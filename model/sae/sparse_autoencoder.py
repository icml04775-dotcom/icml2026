"""
Core Sparse Autoencoder module for DINOv3 patch embeddings.

Based on: "Towards Monosemanticity: Decomposing Language Models
With Dictionary Learning" (Anthropic, 2023)

The SAE learns to decompose dense neural activations into sparse,
interpretable features using an overcomplete dictionary.

Mathematical Formulation:
    Encoder: h = ReLU(W_enc · (z - b_dec) + b_enc)
    Decoder: z' = W_dec · h + b_dec

Where:
    - z ∈ ℝ^D is the input activation (D=1024 for DINOv3 ViT-L)
    - h ∈ ℝ^M is the sparse hidden representation (M=4096 for 4x expansion)
    - W_enc ∈ ℝ^(M×D) is the encoder weight matrix
    - W_dec ∈ ℝ^(D×M) is the decoder weight matrix with unit-norm columns
    - b_enc ∈ ℝ^M is the encoder bias
    - b_dec ∈ ℝ^D is the decoder bias (tied between encoder input and decoder output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for DINOv3 patch embeddings.

    This module implements a sparse autoencoder that decomposes dense
    neural network activations into sparse, interpretable features.

    Architecture:
        Input z (d_input)
            -> subtract b_dec (center)
            -> Linear(d_input, d_hidden) + b_enc
            -> ReLU
            -> sparse h (d_hidden)
            -> Linear(d_hidden, d_input) via W_dec
            -> add b_dec
            -> reconstructed z' (d_input)

    Args:
        d_input: Input dimension (DINOv3 embedding dim, default 1024 for ViT-L)
        d_hidden: Hidden dimension (sparse feature count, default 4096 for 4x expansion)
        l1_coeff: L1 sparsity penalty coefficient (default 1e-4)
        normalize_decoder: Whether to constrain decoder columns to unit L2 norm
        tied_bias: Whether to share bias between encoder preprocessing and decoder output

    Example:
        >>> sae = SparseAutoencoder(d_input=1024, d_hidden=4096)
        >>> z = torch.randn(32, 1024)  # Batch of activations
        >>> output = sae(z)
        >>> print(output['h_sparse'].shape)  # torch.Size([32, 4096])
        >>> print(output['z_reconstructed'].shape)  # torch.Size([32, 1024])
    """

    def __init__(
        self,
        d_input: int = 1024,
        d_hidden: int = 4096,
        l1_coeff: float = 1e-4,
        normalize_decoder: bool = True,
        tied_bias: bool = True,
    ):
        super().__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.normalize_decoder = normalize_decoder
        self.tied_bias = tied_bias

        # Encoder: d_input -> d_hidden
        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        # Decoder: d_hidden -> d_input
        self.W_dec = nn.Parameter(torch.empty(d_input, d_hidden))

        # Decoder bias (optionally tied/shared with encoder preprocessing)
        if tied_bias:
            self.b_dec = nn.Parameter(torch.zeros(d_input))
        else:
            self.register_buffer('b_dec', torch.zeros(d_input))

        # Initialize weights
        self._init_weights()

        # Activation tracking for dead neuron detection
        self.register_buffer('activation_count', torch.zeros(d_hidden))
        self.register_buffer('total_samples', torch.tensor(0.0))

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)

        # Normalize decoder columns to unit norm
        if self.normalize_decoder:
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def normalize_decoder_weights(self):
        """Constrain decoder columns to unit L2 norm.

        This should be called after each optimizer step to maintain
        the unit norm constraint on decoder columns. The decoder columns
        represent "feature directions" in activation space.
        """
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.

        Implements: h = ReLU(W_enc · (z - b_dec) + b_enc)

        Args:
            z: Input activations, shape (*, d_input)

        Returns:
            h: Sparse hidden features, shape (*, d_hidden)
        """
        # Center input (subtract decoder bias)
        z_centered = z - self.b_dec

        # Linear + ReLU
        h = F.relu(F.linear(z_centered, self.W_enc, self.b_enc))

        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Implements: z' = W_dec · h + b_dec

        Args:
            h: Sparse hidden features, shape (*, d_hidden)

        Returns:
            z_reconstructed: Reconstructed activations, shape (*, d_input)
        """
        # W_dec is (d_input, d_hidden), F.linear computes h @ W_dec.T
        # So F.linear(h, W_dec) gives (*, d_hidden) @ (d_hidden, d_input) = (*, d_input)
        z_reconstructed = F.linear(h, self.W_dec) + self.b_dec
        return z_reconstructed

    def forward(
        self,
        z: torch.Tensor,
        return_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SAE.

        Args:
            z: Input activations with shape:
               - (batch_size, d_input) for flat activations
               - (batch_size, n_patches, d_input) for spatial activations
            return_losses: Whether to compute and return losses

        Returns:
            Dictionary containing:
                - z_reconstructed: Reconstructed activations (same shape as input)
                - h_sparse: Sparse hidden features
                - loss_reconstruction: MSE reconstruction loss (if return_losses)
                - loss_sparsity: L1 sparsity penalty (if return_losses)
                - loss_total: Combined loss (if return_losses)
                - sparsity_stats: Dict with activation statistics (if return_losses)
        """
        # Handle different input shapes
        original_shape = z.shape
        if z.dim() == 3:
            B, N, D = z.shape
            z_flat = z.reshape(B * N, D)
        else:
            z_flat = z

        # Encode
        h = self.encode(z_flat)

        # Track activations for dead neuron detection (only during training)
        if self.training:
            with torch.no_grad():
                self.activation_count += (h > 0).float().sum(dim=0)
                self.total_samples += h.shape[0]

        # Decode
        z_reconstructed = self.decode(h)

        # Reshape back if needed
        if z.dim() == 3:
            z_reconstructed = z_reconstructed.reshape(original_shape)
            h = h.reshape(B, N, -1)

        result = {
            'z_reconstructed': z_reconstructed,
            'h_sparse': h,
        }

        if return_losses:
            # Compute losses on flat tensors
            z_flat_for_loss = z.reshape(-1, z.shape[-1]) if z.dim() == 3 else z
            h_flat = h.reshape(-1, h.shape[-1]) if h.dim() == 3 else h
            z_recon_flat = z_reconstructed.reshape(-1, z_reconstructed.shape[-1]) if z_reconstructed.dim() == 3 else z_reconstructed

            loss_reconstruction = F.mse_loss(z_recon_flat, z_flat_for_loss)
            loss_sparsity = h_flat.abs().mean()
            loss_total = loss_reconstruction + self.l1_coeff * loss_sparsity

            # Compute sparsity statistics
            with torch.no_grad():
                num_active = (h_flat > 0).float().sum(dim=-1).mean()
                sparsity_ratio = (h_flat > 0).float().mean()
                max_activation = h_flat.max()

            result.update({
                'loss_reconstruction': loss_reconstruction,
                'loss_sparsity': loss_sparsity,
                'loss_total': loss_total,
                'sparsity_stats': {
                    'num_active_features': num_active,
                    'sparsity_ratio': sparsity_ratio,
                    'max_activation': max_activation,
                }
            })

        return result

    def get_dead_neuron_mask(self, threshold: float = 1e-5) -> torch.Tensor:
        """
        Return mask of neurons that rarely activate.

        A neuron is considered "dead" if its activation frequency is
        below the threshold. Dead neurons waste capacity and should
        be resampled.

        Args:
            threshold: Activation frequency threshold (default 1e-5)

        Returns:
            Boolean tensor of shape (d_hidden,) where True = dead neuron
        """
        if self.total_samples == 0:
            return torch.zeros(self.d_hidden, dtype=torch.bool, device=self.W_enc.device)

        activation_freq = self.activation_count / self.total_samples
        return activation_freq < threshold

    def resample_dead_neurons(
        self,
        data_sample: torch.Tensor,
        threshold: float = 1e-5
    ) -> int:
        """
        Reinitialize dead neurons using data distribution.

        Dead neurons are reinitialized by:
        1. Sampling random activations from the data
        2. Setting encoder weights to normalized versions of these activations
        3. Setting decoder weights to normalized versions (transposed)
        4. Resetting biases to zero

        Args:
            data_sample: Sample of activations to use for reinitialization,
                        shape (n_samples, d_input) or (n_samples, n_patches, d_input)
            threshold: Activation frequency threshold below which neurons are "dead"

        Returns:
            Number of neurons resampled
        """
        dead_mask = self.get_dead_neuron_mask(threshold)
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            return 0

        with torch.no_grad():
            # Ensure data_sample is 2D
            if data_sample.dim() == 3:
                data_sample = data_sample.reshape(-1, data_sample.shape[-1])

            # Sample random directions from data
            num_samples = min(int(num_dead), data_sample.shape[0])
            indices = torch.randperm(data_sample.shape[0], device=data_sample.device)[:num_samples]
            new_directions = data_sample[indices]

            # Handle case where we have fewer samples than dead neurons
            if num_samples < num_dead:
                # Repeat samples to fill
                repeats = (int(num_dead) + num_samples - 1) // num_samples
                new_directions = new_directions.repeat(repeats, 1)[:int(num_dead)]

            # Find dead neuron indices
            dead_indices = torch.where(dead_mask)[0]

            # Reinitialize encoder weights
            self.W_enc.data[dead_indices] = new_directions / (new_directions.norm(dim=-1, keepdim=True) + 1e-8)

            # Reinitialize decoder weights
            self.W_dec.data[:, dead_indices] = F.normalize(new_directions.t(), dim=0)

            # Reset biases
            self.b_enc.data[dead_indices] = 0

            # Reset activation counts
            self.activation_count[dead_indices] = 0

        return int(num_dead)

    def reset_activation_stats(self):
        """Reset activation tracking statistics."""
        self.activation_count.zero_()
        self.total_samples.zero_()

    def get_feature_directions(self) -> torch.Tensor:
        """
        Get the decoder columns (feature directions in activation space).

        Each column of W_dec represents a learned feature direction.
        These can be visualized to understand what each sparse feature represents.

        Returns:
            Tensor of shape (d_hidden, d_input) where each row is a feature direction
        """
        return self.W_dec.t()

    def get_activation_frequencies(self) -> torch.Tensor:
        """
        Get the activation frequency for each feature.

        Returns:
            Tensor of shape (d_hidden,) with frequency values in [0, 1]
        """
        if self.total_samples == 0:
            return torch.zeros(self.d_hidden, device=self.W_enc.device)
        return self.activation_count / self.total_samples

    @property
    def expansion_factor(self) -> float:
        """Return the expansion factor (d_hidden / d_input)."""
        return self.d_hidden / self.d_input

    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, d_hidden={self.d_hidden}, "
            f"expansion={self.expansion_factor:.1f}x, l1_coeff={self.l1_coeff}"
        )
