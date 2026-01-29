"""
TopK Sparse Autoencoder for improved sparsity guarantees.

This module provides a variant of the Sparse Autoencoder that uses TopK
activation selection to guarantee exactly K features are active per token,
rather than relying solely on L1 regularization.

Key improvements over soft L1 sparsity:
1. Guaranteed sparsity: Exactly K features active (not "approximately sparse")
2. Better feature utilization: Prevents dead neurons more effectively
3. More interpretable: Consistent number of active features across inputs
4. Simpler tuning: Just set K, no need to tune L1 coefficient

Extended features:
- Multi-layer support: Concatenate features from multiple ViT layers
- Unified dictionary: Single W_dec spans all layer features
- Steering API: Built-in support for feature manipulation
- Error-corrected steering: Preserve reconstruction error during interventions

Based on:
- "Scaling Monosemanticity" (Anthropic, 2024)
- "TopK Sparse Autoencoders" research
- "Interpretable and Testable Vision Features via SAEs" (Stevens et al., 2025)

Usage:
    # Single-layer mode (original)
    >>> sae = TopKSparseAutoencoder(d_input=1024, d_hidden=4096, k=32, n_layers=1)
    >>> output = sae(activations)
    
    # Multi-layer mode (unified dictionary)
    >>> sae = TopKSparseAutoencoder(d_input=1024, d_hidden=16384, k=64, n_layers=4)
    >>> layer_features = [layer4_feat, layer11_feat, layer17_feat, layer23_feat]
    >>> output = sae(layer_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any


class TopKSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with TopK activation selection.
    
    Supports unified dictionary across multiple ViT layers.
    Features from all layers are concatenated before encoding.
    
    Architecture:
        Input z (n_layers × d_input concatenated)
            -> subtract b_dec (center)
            -> Linear(d_concat, d_hidden) + b_enc
            -> ReLU
            -> TopK selection (keep top K activations)
            -> sparse h (d_hidden) with exactly K non-zeros
            -> Linear(d_hidden, d_concat) via W_dec
            -> add b_dec
            -> reconstructed z' (d_concat)
            -> split back to per-layer features
    
    Args:
        d_input: Per-layer input dimension (768, 1024, 1536)
        d_hidden: Dictionary size (sparse feature count)
        k: Number of features to keep active per token
        n_layers: Number of ViT layers to concatenate (1 = single-layer mode)
        layer_weights: Optional per-layer reconstruction weights
        aux_k: Number of top features for auxiliary loss
        dead_threshold: Threshold for considering a neuron dead
        normalize_decoder: Whether to constrain decoder columns to unit L2 norm
        tied_bias: Whether to share bias between encoder preprocessing and decoder
    
    Example:
        # Multi-layer unified dictionary
        >>> sae = TopKSparseAutoencoder(d_input=1024, d_hidden=16384, k=64, n_layers=4)
        >>> layer_features = [torch.randn(2, 256, 1024) for _ in range(4)]
        >>> output = sae(layer_features)
        >>> print(output['h_sparse'].shape)  # torch.Size([2, 256, 16384])
        >>> print(len(output['layer_reconstructed']))  # 4
    """
    
    def __init__(
        self,
        d_input: int = 1024,
        d_hidden: int = 16384,
        k: int = 64,
        n_layers: int = 4,
        layer_weights: Optional[List[float]] = None,
        aux_k: Optional[int] = None,
        aux_loss_coeff: float = 0.1,  # Coefficient for auxiliary loss (dead neuron penalty)
        dead_threshold: float = 1e-5,
        normalize_decoder: bool = True,
        tied_bias: bool = True,
        # Orthogonality parameters (OrtSAE)
        ortho_coeff: float = 0.0,
        ortho_method: str = "sampled",
        ortho_sample_size: int = 1024,
        ortho_compute_freq: int = 1,
    ):
        super().__init__()
        
        # Store config
        self.d_input = d_input
        self.n_layers = n_layers
        self.d_concat = d_input * n_layers  # Total input dimension
        self.d_hidden = d_hidden
        self.k = k
        self.aux_k = aux_k if aux_k is not None else min(k * 2, d_hidden)
        self.aux_loss_coeff = aux_loss_coeff
        self.dead_threshold = dead_threshold
        self.normalize_decoder = normalize_decoder
        self.tied_bias = tied_bias
        
        # Orthogonality config (OrtSAE)
        self.ortho_coeff = ortho_coeff
        self.ortho_method = ortho_method
        self.ortho_sample_size = ortho_sample_size
        self.ortho_compute_freq = ortho_compute_freq
        
        # For compatibility with existing code that checks l1_coeff
        self.l1_coeff = 0.0  # Not used in TopK, but kept for interface compatibility
        
        # Layer weights for reconstruction loss
        if layer_weights is None:
            layer_weights = [1.0] * n_layers
        self.register_buffer('layer_weights', torch.tensor(layer_weights))
        
        # Encoder: d_concat -> d_hidden
        self.W_enc = nn.Parameter(torch.empty(d_hidden, self.d_concat))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        
        # Decoder: d_hidden -> d_concat
        self.W_dec = nn.Parameter(torch.empty(self.d_concat, d_hidden))
        
        # Decoder bias (optionally tied/shared with encoder preprocessing)
        if tied_bias:
            self.b_dec = nn.Parameter(torch.zeros(self.d_concat))
        else:
            self.register_buffer('b_dec', torch.zeros(self.d_concat))
        
        # Initialize weights
        self._init_weights()
        
        # Activation tracking for dead neuron detection
        self.register_buffer('activation_count', torch.zeros(d_hidden))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # Step counter for orthogonality loss frequency
        self.register_buffer('ortho_step', torch.tensor(0))
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)
        
        # Normalize decoder columns to unit norm
        if self.normalize_decoder:
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
    
    def normalize_decoder_weights(self):
        """Constrain decoder columns to unit L2 norm."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
    
    # =========================================================================
    # LAYER INDEXING
    # =========================================================================
    
    def get_layer_slice(self, layer_idx: int) -> slice:
        """
        Get slice for specific layer's dimensions in concatenated space.
        
        Args:
            layer_idx: Index of the layer (0 to n_layers-1)
        
        Returns:
            Slice object for indexing into concatenated dimension
        """
        start = layer_idx * self.d_input
        end = (layer_idx + 1) * self.d_input
        return slice(start, end)
    
    def split_layers(self, z_concat: torch.Tensor) -> List[torch.Tensor]:
        """
        Split concatenated features back into per-layer tensors.
        
        Args:
            z_concat: Concatenated features (..., n_layers × d_input)
        
        Returns:
            List of per-layer tensors, each (..., d_input)
        """
        layers = []
        for i in range(self.n_layers):
            layer_slice = self.get_layer_slice(i)
            layers.append(z_concat[..., layer_slice])
        return layers
    
    # =========================================================================
    # ENCODING
    # =========================================================================
    
    def encode(
        self,
        z: Union[torch.Tensor, List[torch.Tensor]],
        apply_topk: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input activations to sparse features with TopK selection.
        
        Supports both single tensor input (for single-layer or pre-concatenated)
        and list of tensors (for multi-layer mode).
        
        Args:
            z: Input activations:
               - Single tensor: (*, d_concat) or (*, d_input) for n_layers=1
               - List of tensors: [(*, d_input)] × n_layers
            apply_topk: Whether to apply TopK selection
        
        Returns:
            If z is a list: (h_sparse, z_concat) tuple
            If z is a tensor: h_sparse tensor only
        """
        # Handle list input (multi-layer mode)
        if isinstance(z, list):
            assert len(z) == self.n_layers, \
                f"Expected {self.n_layers} layers, got {len(z)}"
            z_concat = torch.cat(z, dim=-1)  # (*, n_layers × d_input)
            return_concat = True
        else:
            z_concat = z
            return_concat = False
        
        # Center input (subtract decoder bias)
        z_centered = z_concat - self.b_dec
        
        # Linear projection
        pre_activations = F.linear(z_centered, self.W_enc, self.b_enc)
        
        # ReLU activation
        h = F.relu(pre_activations)
        
        if apply_topk:
            # TopK selection: keep only top K activations per sample
            h = self._apply_topk(h)
        
        if return_concat:
            return h, z_concat
        return h
    
    def _apply_topk(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply TopK selection to keep only the top K activations.
        
        This ensures exactly K features are active per token.
        Uses straight-through estimator for gradient flow.
        
        Args:
            h: Pre-TopK activations, shape (*, d_hidden)
        
        Returns:
            h_sparse: Post-TopK activations with exactly K non-zeros per sample
        """
        original_shape = h.shape
        h_flat = h.reshape(-1, self.d_hidden)
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(h_flat, self.k, dim=-1)
        
        # Create sparse output
        h_sparse = torch.zeros_like(h_flat)
        h_sparse.scatter_(-1, topk_indices, topk_values)
        
        # Straight-through estimator: gradients flow through as if no TopK was applied
        # but forward pass has TopK selection
        h_sparse = h_flat - h_flat.detach() + h_sparse.detach()
        
        return h_sparse.reshape(original_shape)
    
    # =========================================================================
    # DECODING
    # =========================================================================
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.
        
        Args:
            h: Sparse hidden features, shape (*, d_hidden)
        
        Returns:
            z_reconstructed: Reconstructed activations, shape (*, d_concat)
        """
        z_reconstructed = F.linear(h, self.W_dec) + self.b_dec
        return z_reconstructed
    
    # =========================================================================
    # FORWARD PASS
    # =========================================================================
    
    def forward(
        self,
        z: Union[torch.Tensor, List[torch.Tensor]],
        return_losses: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through TopK SAE.
        
        Args:
            z: Input activations:
               - Single tensor: (batch_size, [n_patches,] d_input) for n_layers=1
               - List of tensors: [(batch_size, n_patches, d_input)] × n_layers
            return_losses: Whether to compute and return losses
        
        Returns:
            Dictionary containing:
                - z_reconstructed: Reconstructed activations (concatenated)
                - h_sparse: Sparse hidden features (exactly K non-zeros per sample)
                - layer_reconstructed: List of per-layer reconstructed features
                - z_concat: Concatenated input (for multi-layer mode)
                - loss_reconstruction: MSE reconstruction loss (if return_losses)
                - loss_auxiliary: Auxiliary loss for dead neurons (if return_losses)
                - loss_total: Combined loss (if return_losses)
                - sparsity_stats: Dict with activation statistics (if return_losses)
        """
        # Handle input format
        is_list_input = isinstance(z, list)
        
        if is_list_input:
            # Multi-layer mode: encode returns (h, z_concat)
            h, z_concat = self.encode(z, apply_topk=True)
            original_layers = z
        else:
            # Single tensor mode
            h = self.encode(z, apply_topk=True)
            z_concat = z
            original_layers = self.split_layers(z_concat) if self.n_layers > 1 else [z]
        
        # Handle different input shapes for tracking
        if h.dim() == 3:
            B, N, D = h.shape
            h_flat = h.reshape(B * N, D)
        else:
            h_flat = h
        
        # Track activations for dead neuron detection (only during training)
        if self.training:
            with torch.no_grad():
                self.activation_count += (h_flat > 0).float().sum(dim=0)
                self.total_samples += h_flat.shape[0]
        
        # Decode
        z_reconstructed = self.decode(h)
        
        # Split back to per-layer
        layer_reconstructed = self.split_layers(z_reconstructed)
        
        result = {
            'z_reconstructed': z_reconstructed,
            'h_sparse': h,
            'layer_reconstructed': layer_reconstructed,
            'z_concat': z_concat,
        }
        
        if return_losses:
            # Compute losses on flat tensors
            z_flat = z_concat.reshape(-1, self.d_concat) if z_concat.dim() == 3 else z_concat
            h_flat_loss = h.reshape(-1, self.d_hidden) if h.dim() == 3 else h
            
            # Per-layer reconstruction loss (weighted)
            loss_recon = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            for i, (orig, recon) in enumerate(zip(original_layers, layer_reconstructed)):
                layer_loss = F.mse_loss(recon, orig)
                loss_recon = loss_recon + self.layer_weights[i] * layer_loss
            loss_recon = loss_recon / self.layer_weights.sum()
            
            # Auxiliary loss for dead neurons
            loss_auxiliary = self._compute_auxiliary_loss(z_flat, h_flat_loss)
            
            # Orthogonality loss (OrtSAE)
            loss_ortho = self.compute_ortho_loss()
            
            # Total loss (no L1 term for TopK)
            loss_total = loss_recon + loss_auxiliary + loss_ortho
            
            # Compute sparsity statistics
            with torch.no_grad():
                num_active = (h_flat_loss > 0).float().sum(dim=-1).mean()
                sparsity_ratio = (h_flat_loss > 0).float().mean()
                max_activation = h_flat_loss.max()
            
            # For compatibility with soft L1 SAE, include a "sparsity loss" term
            # that's just for logging (not used in training)
            loss_sparsity = h_flat_loss.abs().mean()
            
            result.update({
                'loss_reconstruction': loss_recon,
                'loss_sparsity': loss_sparsity,  # For logging compatibility
                'loss_auxiliary': loss_auxiliary,
                'loss_orthogonality': loss_ortho,  # OrtSAE loss
                'loss_total': loss_total,
                'sparsity_stats': {
                    'num_active_features': num_active,
                    'sparsity_ratio': sparsity_ratio,
                    'max_activation': max_activation,
                }
            })
        
        return result
    
    # =========================================================================
    # ORTHOGONALITY LOSS (OrtSAE)
    # =========================================================================
    
    def _compute_ortho_loss_full(self) -> torch.Tensor:
        """
        Compute full orthogonality loss (exact but expensive).
        
        Penalizes high cosine similarity between decoder columns.
        
        Complexity: O(d_concat × d_hidden²)
        Memory: O(d_hidden²)
        
        Returns:
            Orthogonality loss scalar (unweighted)
        """
        # Normalize decoder columns to unit length
        W_dec_norm = F.normalize(self.W_dec, dim=0)  # (d_concat, d_hidden)
        
        # Full cosine similarity matrix
        cos_sim = W_dec_norm.T @ W_dec_norm  # (d_hidden, d_hidden)
        
        # Zero diagonal (self-similarity is always 1)
        cos_sim = cos_sim - torch.diag(cos_sim.diag())
        
        # Squared loss (stronger gradients for highly similar features)
        ortho_loss = (cos_sim ** 2).mean()
        
        return ortho_loss
    
    def _compute_ortho_loss_sampled(self) -> torch.Tensor:
        """
        Compute sampled orthogonality loss (efficient approximation).
        
        Randomly samples features to compute similarity.
        
        Complexity: O(d_concat × sample_size²)
        Memory: O(sample_size²)
        
        Returns:
            Orthogonality loss scalar (unweighted)
        """
        # Sample random feature indices
        sample_size = min(self.ortho_sample_size, self.d_hidden)
        indices = torch.randperm(self.d_hidden, device=self.W_dec.device)[:sample_size]
        
        # Extract sampled decoder columns
        W_dec_sampled = self.W_dec[:, indices]  # (d_concat, sample_size)
        W_dec_norm = F.normalize(W_dec_sampled, dim=0)
        
        # Similarity among sampled features
        cos_sim = W_dec_norm.T @ W_dec_norm  # (sample_size, sample_size)
        cos_sim = cos_sim - torch.diag(cos_sim.diag())
        
        ortho_loss = (cos_sim ** 2).mean()
        
        return ortho_loss
    
    def _compute_ortho_loss_chunked(self, chunk_size: int = 2048) -> torch.Tensor:
        """
        Compute orthogonality loss in chunks (memory efficient).
        
        Processes similarity matrix in blocks to avoid OOM.
        
        Complexity: O(d_concat × d_hidden²)
        Memory: O(chunk_size × d_hidden)
        
        Args:
            chunk_size: Size of chunks for processing
        
        Returns:
            Orthogonality loss scalar (unweighted)
        """
        W_dec_norm = F.normalize(self.W_dec, dim=0)  # (d_concat, d_hidden)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(0, self.d_hidden, chunk_size):
            chunk_i = W_dec_norm[:, i:i+chunk_size]  # (d_concat, chunk_size)
            
            for j in range(i, self.d_hidden, chunk_size):
                chunk_j = W_dec_norm[:, j:j+chunk_size]  # (d_concat, chunk_size)
                
                # Similarity between chunks
                cos_sim = chunk_i.T @ chunk_j  # (chunk_i_size, chunk_j_size)
                
                # Handle diagonal for same chunk
                if i == j:
                    cos_sim = cos_sim - torch.diag(cos_sim.diag())
                    num_elements = cos_sim.numel() - cos_sim.shape[0]
                else:
                    num_elements = cos_sim.numel()
                
                total_loss = total_loss + (cos_sim ** 2).sum()
                num_pairs += num_elements
        
        return total_loss / max(num_pairs, 1)
    
    def compute_ortho_loss(self) -> torch.Tensor:
        """
        Compute orthogonality loss using configured method.
        
        Returns:
            Weighted orthogonality loss (or 0 if disabled/skipped)
        """
        if self.ortho_coeff == 0:
            return torch.tensor(0.0, device=self.W_dec.device, dtype=self.W_dec.dtype)
        
        # Frequency-based computation (skip some steps if configured)
        if self.training:
            self.ortho_step += 1
            if self.ortho_compute_freq > 1 and self.ortho_step % self.ortho_compute_freq != 0:
                return torch.tensor(0.0, device=self.W_dec.device, dtype=self.W_dec.dtype)
        
        # Compute based on method
        if self.ortho_method == "full":
            ortho_loss = self._compute_ortho_loss_full()
        elif self.ortho_method == "sampled":
            ortho_loss = self._compute_ortho_loss_sampled()
        elif self.ortho_method == "chunked":
            ortho_loss = self._compute_ortho_loss_chunked()
        else:
            raise ValueError(f"Unknown ortho_method: {self.ortho_method}. "
                           f"Supported: 'full', 'sampled', 'chunked'")
        
        return self.ortho_coeff * ortho_loss
    
    def compute_ortho_stats(self) -> Dict[str, torch.Tensor]:
        """
        Compute orthogonality statistics for monitoring.
        
        Returns:
            Dict with statistics about feature orthogonality:
                - ortho_mean_sim: Mean absolute cosine similarity
                - ortho_max_sim: Max absolute cosine similarity
                - ortho_std_sim: Std of cosine similarities
                - ortho_high_sim_ratio: Ratio of pairs with |sim| > 0.5
        """
        with torch.no_grad():
            W_dec_norm = F.normalize(self.W_dec, dim=0)
            
            # Sample for efficiency
            sample_size = min(1024, self.d_hidden)
            indices = torch.randperm(self.d_hidden, device=self.W_dec.device)[:sample_size]
            W_sampled = W_dec_norm[:, indices]
            
            cos_sim = W_sampled.T @ W_sampled
            cos_sim.fill_diagonal_(0)
            cos_sim_abs = cos_sim.abs()
            
            return {
                'ortho_mean_sim': cos_sim_abs.mean(),
                'ortho_max_sim': cos_sim_abs.max(),
                'ortho_std_sim': cos_sim_abs.std(),
                'ortho_high_sim_ratio': (cos_sim_abs > 0.5).float().mean(),
            }
    
    def _compute_auxiliary_loss(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to prevent dead neurons (properly normalized).
        
        This loss encourages neurons that aren't in the top-K to still
        have non-zero pre-activations, preventing them from becoming
        completely dead.
        
        The loss is normalized to have magnitude similar to reconstruction loss:
        - Normalized by number of dead features
        - Scaled by aux_loss_coeff (default 0.1)
        
        Args:
            z: Input activations (B, d_concat)
            h: Post-TopK sparse activations (B, d_hidden)
        
        Returns:
            Auxiliary loss scalar (normalized)
        """
        # Get pre-activations (before TopK)
        z_centered = z - self.b_dec
        pre_activations = F.relu(F.linear(z_centered, self.W_enc, self.b_enc))
        
        # For features NOT in top-K, we want them to still have some activation
        # Get the aux_k-th largest activation as threshold
        topk_aux_values, _ = torch.topk(pre_activations, self.aux_k, dim=-1)
        threshold = topk_aux_values[:, -1:]  # (B, 1) - smallest of top aux_k
        
        # Dead features: those below the aux_k threshold
        dead_mask = pre_activations < threshold
        
        # Auxiliary loss: push dead features toward threshold
        # PROPERLY NORMALIZED: mean over features, not sum
        aux_loss = (threshold - pre_activations).clamp(min=0)
        
        # Normalize by the number of dead features to keep scale similar to recon loss
        num_dead = dead_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        aux_loss = (aux_loss * dead_mask.float()) / num_dead  # Normalize per-sample
        aux_loss = aux_loss.sum(dim=-1).mean()  # Mean over batch
        
        return self.aux_loss_coeff * aux_loss
    
    # =========================================================================
    # STEERING API
    # =========================================================================
    
    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """
        Get the full dictionary direction for a feature.
        
        Args:
            feature_idx: Index of the feature (0 to d_hidden-1)
        
        Returns:
            Feature direction vector (d_concat,)
        """
        return self.W_dec[:, feature_idx]
    
    def get_feature_layer_direction(
        self,
        feature_idx: int,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Get the dictionary direction for a specific feature and layer.
        
        Args:
            feature_idx: Index of the feature
            layer_idx: Index of the layer
        
        Returns:
            Feature direction for that layer (d_input,)
        """
        layer_slice = self.get_layer_slice(layer_idx)
        return self.W_dec[layer_slice, feature_idx]
    
    def steer_features(
        self,
        h_sparse: torch.Tensor,
        interventions: Dict[int, Tuple[str, float]]
    ) -> torch.Tensor:
        """
        Apply steering interventions to sparse features.
        
        Args:
            h_sparse: Sparse features (*, d_hidden)
            interventions: {feature_idx: (operation, value)}
                          operations: 'boost', 'suppress', 'clamp'
        
        Returns:
            Modified sparse features
        
        Example:
            >>> interventions = {
            ...     100: ('boost', 5.0),    # Add 5.0 to feature 100
            ...     200: ('suppress', 0),   # Zero out feature 200
            ...     300: ('clamp', 3.0),    # Set feature 300 to exactly 3.0
            ... }
            >>> h_steered = sae.steer_features(h_sparse, interventions)
        """
        h_modified = h_sparse.clone()
        
        for feature_idx, (operation, value) in interventions.items():
            if operation == 'boost':
                h_modified[..., feature_idx] = h_modified[..., feature_idx] + value
            elif operation == 'suppress':
                h_modified[..., feature_idx] = 0.0
            elif operation == 'clamp':
                h_modified[..., feature_idx] = value
            else:
                raise ValueError(f"Unknown operation: {operation}. "
                               f"Supported: 'boost', 'suppress', 'clamp'")
        
        return h_modified
    
    def error_corrected_steering(
        self,
        z: Union[torch.Tensor, List[torch.Tensor]],
        interventions: Dict[int, Tuple[str, float]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Apply error-corrected steering (Stevens et al. approach).
        
        Preserves reconstruction error while applying interventions.
        This ensures the intervention effect is isolated from SAE reconstruction loss.
        
        Formula: modified_x = error + modified_x̂
                           = (x - x̂) + decode(steer(encode(x)))
                           = x + Δ  where Δ = modified_x̂ - x̂
        
        Args:
            z: Original features (tensor or list of tensors)
            interventions: Feature interventions to apply
        
        Returns:
            Modified features with error correction
            - If input was list: returns list of per-layer tensors
            - If input was tensor: returns tensor
        
        Example:
            >>> interventions = {100: ('suppress', 0)}  # Suppress feature 100
            >>> steered_layers = sae.error_corrected_steering(layer_features, interventions)
        """
        is_list_input = isinstance(z, list)
        
        # Get original reconstruction
        if is_list_input:
            h_sparse, z_concat = self.encode(z, apply_topk=True)
        else:
            h_sparse = self.encode(z, apply_topk=True)
            z_concat = z
        
        z_recon = self.decode(h_sparse)
        
        # Compute reconstruction error
        error = z_concat - z_recon
        
        # Apply interventions
        h_modified = self.steer_features(h_sparse, interventions)
        
        # Decode modified features
        z_modified_recon = self.decode(h_modified)
        
        # Error-corrected output: error + modified_recon
        z_steered = error + z_modified_recon
        
        # Return in same format as input
        if is_list_input:
            return self.split_layers(z_steered)
        return z_steered
    
    # =========================================================================
    # DEAD NEURON HANDLING
    # =========================================================================
    
    def get_dead_neuron_mask(self, threshold: Optional[float] = None) -> torch.Tensor:
        """
        Return mask of neurons that rarely activate.
        
        Args:
            threshold: Activation frequency threshold (default uses self.dead_threshold)
        
        Returns:
            Boolean tensor of shape (d_hidden,) where True = dead neuron
        """
        if threshold is None:
            threshold = self.dead_threshold
        
        if self.total_samples == 0:
            return torch.zeros(self.d_hidden, dtype=torch.bool, device=self.W_enc.device)
        
        activation_freq = self.activation_count / self.total_samples
        return activation_freq < threshold
    
    def resample_dead_neurons(
        self,
        data_sample: Union[torch.Tensor, List[torch.Tensor]],
        threshold: Optional[float] = None
    ) -> int:
        """
        Reinitialize dead neurons using data distribution.
        
        Args:
            data_sample: Sample of activations for reinitialization
                        (tensor or list of tensors)
            threshold: Activation frequency threshold
        
        Returns:
            Number of neurons resampled
        """
        dead_mask = self.get_dead_neuron_mask(threshold)
        num_dead = dead_mask.sum().item()
        
        if num_dead == 0:
            return 0
        
        with torch.no_grad():
            # Handle list input
            if isinstance(data_sample, list):
                data_sample = torch.cat(data_sample, dim=-1)
            
            # Ensure data_sample is 2D
            if data_sample.dim() == 3:
                data_sample = data_sample.reshape(-1, data_sample.shape[-1])
            
            # Ensure data_sample is on the same device as SAE weights
            target_device = self.W_enc.device
            data_sample = data_sample.to(target_device)
            
            # Sample random directions from data
            num_samples = min(int(num_dead), data_sample.shape[0])
            indices = torch.randperm(data_sample.shape[0], device=target_device)[:num_samples]
            new_directions = data_sample[indices]
            
            # Handle case where we have fewer samples than dead neurons
            if num_samples < num_dead:
                repeats = (int(num_dead) + num_samples - 1) // num_samples
                new_directions = new_directions.repeat(repeats, 1)[:int(num_dead)]
            
            # Find dead neuron indices (ensure on same device)
            dead_indices = torch.where(dead_mask)[0].to(target_device)
            
            # Reinitialize encoder weights (normalized)
            self.W_enc.data[dead_indices] = F.normalize(new_directions, dim=-1)
            
            # Reinitialize decoder weights (normalized)
            self.W_dec.data[:, dead_indices] = F.normalize(new_directions.t(), dim=0)
            
            # Reset biases to small values
            self.b_enc.data[dead_indices] = 0.0
            
            # Reset activation counts
            self.activation_count[dead_indices] = 0
        
        return int(num_dead)
    
    def reset_activation_stats(self):
        """Reset activation tracking statistics."""
        self.activation_count.zero_()
        self.total_samples.zero_()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_feature_directions(self) -> torch.Tensor:
        """Get the decoder columns (feature directions in activation space)."""
        return self.W_dec.t()
    
    def get_activation_frequencies(self) -> torch.Tensor:
        """Get the activation frequency for each feature."""
        if self.total_samples == 0:
            return torch.zeros(self.d_hidden, device=self.W_enc.device)
        return self.activation_count / self.total_samples
    
    def find_active_features(
        self,
        z: Union[torch.Tensor, List[torch.Tensor]],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find the most active features for a given input.
        
        Args:
            z: Input activations (tensor or list of tensors)
            top_k: Number of top features to return
        
        Returns:
            List of (feature_idx, mean_activation) tuples
        """
        with torch.no_grad():
            if isinstance(z, list):
                h, _ = self.encode(z, apply_topk=False)  # Get pre-topk activations
            else:
                h = self.encode(z, apply_topk=False)
            
            # Average across batch and spatial dimensions
            if h.dim() == 3:
                h_mean = h.mean(dim=(0, 1))
            else:
                h_mean = h.mean(dim=0)
            
            # Get top-k
            values, indices = torch.topk(h_mean, top_k)
            
            return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
    
    @property
    def expansion_factor(self) -> float:
        """Return the expansion factor (d_hidden / d_concat)."""
        return self.d_hidden / self.d_concat
    
    def extra_repr(self) -> str:
        ortho_str = f", ortho={self.ortho_coeff}" if self.ortho_coeff > 0 else ""
        return (
            f"d_input={self.d_input}, d_hidden={self.d_hidden}, "
            f"k={self.k}, n_layers={self.n_layers}, "
            f"d_concat={self.d_concat}, expansion={self.expansion_factor:.1f}x, "
            f"aux_coeff={self.aux_loss_coeff}"
            f"{ortho_str}"
        )


class JumpReLUSparseAutoencoder(TopKSparseAutoencoder):
    """
    Sparse Autoencoder with JumpReLU activation.
    
    JumpReLU is an alternative to TopK that uses a learnable threshold
    per feature instead of a fixed K. This can provide adaptive sparsity
    while still guaranteeing sparsity.
    
    JumpReLU(x) = x if x > θ else 0
    
    where θ is a learnable threshold per feature.
    
    Args:
        d_input: Input dimension
        d_hidden: Hidden dimension
        initial_threshold: Initial threshold value (default 0.1)
        n_layers: Number of ViT layers
        **kwargs: Additional arguments passed to TopKSparseAutoencoder
    
    Note: This is an experimental alternative to TopK.
    """
    
    def __init__(
        self,
        d_input: int = 1024,
        d_hidden: int = 4096,
        initial_threshold: float = 0.1,
        n_layers: int = 1,
        **kwargs
    ):
        # Remove k from kwargs if present since JumpReLU doesn't use it
        kwargs.pop('k', None)
        super().__init__(
            d_input=d_input,
            d_hidden=d_hidden,
            k=32,  # Placeholder, not used
            n_layers=n_layers,
            **kwargs
        )
        
        # Learnable threshold per feature
        self.threshold = nn.Parameter(torch.full((d_hidden,), initial_threshold))
    
    def encode(
        self,
        z: Union[torch.Tensor, List[torch.Tensor]],
        apply_topk: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode with JumpReLU activation instead of TopK.
        
        Args:
            z: Input activations (tensor or list)
            apply_topk: Ignored, JumpReLU always applies thresholding
        
        Returns:
            Sparse features with JumpReLU activation
        """
        # Handle list input
        if isinstance(z, list):
            z_concat = torch.cat(z, dim=-1)
            return_concat = True
        else:
            z_concat = z
            return_concat = False
        
        z_centered = z_concat - self.b_dec
        pre_activations = F.linear(z_centered, self.W_enc, self.b_enc)
        
        # JumpReLU: x if x > threshold else 0
        # Use straight-through estimator for gradients
        h = F.relu(pre_activations)
        mask = (h > self.threshold.unsqueeze(0)).float()
        
        # Straight-through: gradient flows through as if no thresholding
        h_sparse = h * mask + h.detach() * (1 - mask) - h.detach()
        
        if return_concat:
            return h_sparse, z_concat
        return h_sparse
    
    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, d_hidden={self.d_hidden}, "
            f"n_layers={self.n_layers}, d_concat={self.d_concat}, "
            f"expansion={self.expansion_factor:.1f}x, "
            f"mean_threshold={self.threshold.mean().item():.4f}"
        )
