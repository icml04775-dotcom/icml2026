"""
DINOv3 + TopK SAE Integrated Segmentation Models.

This module provides segmentation models that integrate TopK Sparse Autoencoders
with DINOv3 backbones for improved interpretability with guaranteed sparsity.

Key features:
- Uses TopKSparseAutoencoder with unified dictionary across multiple ViT layers
- Guarantees exactly K features active per token (not "approximately sparse")
- Supports steering via feature manipulation
- Error-corrected steering preserves reconstruction error

Models:
    - Dinov3TopKSAEDPTSegmentation: TopK SAE with DPT decoder
    - Dinov3TopKSAEUNetSegmentation: TopK SAE with UNet decoder

Based on:
- "Scaling Monosemanticity" (Anthropic, 2024)
- "Interpretable and Testable Vision Features via SAEs" (Stevens et al., 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
from dataclasses import dataclass

from .sae import TopKSparseAutoencoder
from .dinov3_model import (
    Dinov3Backbone,
    Dinov3SegmentationHead,
    Dinov3UNetDecoder,
)
from .depth_dpt import DPTHead
from .dinov3_sae_model import SAEIntermediateFeatures, SAEModelMixin


class Dinov3TopKSAEDPTSegmentation(SAEModelMixin, nn.Module):
    """
    DINOv3 + TopK SAE with DPT (Dense Prediction Transformer) decoder.
    
    Uses unified dictionary across all backbone layers.
    Features from all layers are concatenated before SAE encoding.
    
    Key parameters:
        - sae_k: Number of features to keep active per token (default 64)
        - n_backbone_layers: Number of backbone layers (default 4)
        - use_error_correction: Whether to use error-corrected steering (default True)
    
    Output Modes:
        - "inference": Returns logits tensor only
        - "train": Returns dict with logits, sae_loss, sae_stats
        - "debug": Returns SAEIntermediateFeatures with all intermediate outputs
    
    Args:
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type ('vitb', 'vitl', 'vitg')
        dataset: Dataset type for backbone ('lvd1689m' or 'sat493m')
        sae_hidden_dim: SAE hidden dimension / dictionary size (default 16384)
        sae_k: Number of active features per token (default 64)
        n_backbone_layers: Number of backbone layers to use (default 4)
        use_sparse_features: Use sparse or reconstructed features for decoder
        freeze_backbone: Freeze backbone weights during training
        load_pretrained_backbone: Whether to load pretrained backbone
        patch_size: ViT patch size (default 16)
        use_auxiliary: Use auxiliary outputs for deep supervision
        use_error_correction: Use error-corrected steering (default True)
    
    Example:
        >>> model = Dinov3TopKSAEDPTSegmentation(
        ...     num_classes=9,
        ...     backbone_type="vitl",
        ...     sae_hidden_dim=16384,
        ...     sae_k=64,
        ...     n_backbone_layers=4
        ... )
        >>> output = model(images)
        >>> # Steering example
        >>> interventions = {100: ('suppress', 0)}  # Suppress feature 100
        >>> steered_logits = model.steer(images, interventions)
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        sae_hidden_dim: int = 16384,
        sae_k: int = 64,
        n_backbone_layers: int = 4,
        use_sparse_features: bool = False,
        freeze_backbone: bool = True,
        load_pretrained_backbone: bool = False,
        patch_size: int = 16,
        use_auxiliary: bool = False,
        use_error_correction: bool = True,
        # SAE loss coefficients
        sae_aux_loss_coeff: float = 0.1,  # Coefficient for auxiliary (dead neuron) loss
        # Orthogonality parameters (OrtSAE)
        sae_ortho_coeff: float = 0.0,
        sae_ortho_method: str = "sampled",
        sae_ortho_sample_size: int = 1024,
        sae_ortho_compute_freq: int = 1,
    ):
        super().__init__()
        self._init_sae_mixin()
        
        self.num_classes = num_classes
        self.use_sparse_features = use_sparse_features
        self.freeze_backbone = freeze_backbone
        self.patch_size = patch_size
        self.sae_hidden_dim = sae_hidden_dim
        self.sae_k = sae_k
        self.n_backbone_layers = n_backbone_layers
        self.use_error_correction = use_error_correction
        
        # Embedding dimensions
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        self.d_input = embedding_dims[backbone_type]
        
        # Backbone (reshape=False for DPT compatibility)
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=False,
            return_class_token=True,
            load_pretrained_backbone=load_pretrained_backbone
        )
        
        # Unified Multi-Layer TopK Sparse Autoencoder with OrtSAE support
        self.sae = TopKSparseAutoencoder(
            d_input=self.d_input,
            d_hidden=sae_hidden_dim,
            k=sae_k,
            n_layers=n_backbone_layers,
            aux_loss_coeff=sae_aux_loss_coeff,
            ortho_coeff=sae_ortho_coeff,
            ortho_method=sae_ortho_method,
            ortho_sample_size=sae_ortho_sample_size,
            ortho_compute_freq=sae_ortho_compute_freq,
        )
        
        # Determine decoder input dimension
        if use_sparse_features:
            decoder_in_dim = sae_hidden_dim
        else:
            decoder_in_dim = self.d_input
        
        # DPT Head
        self.segmentation_head = DPTHead(
            decoder_in_dim,
            num_classes=num_classes,
            use_auxiliary=use_auxiliary,
            patch_size=patch_size
        )
        
        # Projection layer for sparse features to decoder (if needed)
        if use_sparse_features:
            self.sparse_to_decoder = nn.ModuleList([
                nn.Linear(sae_hidden_dim, decoder_in_dim)
                for _ in range(n_backbone_layers)
            ])
    
    def _extract_layer_features(
        self,
        backbone_features: List
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Extract patch tokens and class tokens from backbone features.
        
        Args:
            backbone_features: List of backbone outputs (may be tuples)
        
        Returns:
            layer_features: List of patch token tensors
            class_tokens: List of class token tensors (or None)
        """
        layer_features = []
        class_tokens = []
        
        for feat in backbone_features:
            if isinstance(feat, tuple):
                patch_tokens, class_token = feat
                layer_features.append(patch_tokens)
                class_tokens.append(class_token)
            else:
                layer_features.append(feat)
                class_tokens.append(None)
        
        return layer_features, class_tokens
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor, List, List, List]:
        """Internal forward implementation with unified multi-layer SAE."""
        B, C, H, W = x.shape
        
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image size {H}x{W} not divisible by patch size {self.patch_size}"
            )
        
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # Extract backbone features
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_features = self.backbone(x)
        else:
            backbone_features = self.backbone(x)
        
        # Store backbone features if needed
        if return_intermediates:
            backbone_features_stored = []
            for feat in backbone_features:
                if isinstance(feat, tuple):
                    backbone_features_stored.append(
                        (feat[0].detach().clone(), feat[1].detach().clone())
                    )
                else:
                    backbone_features_stored.append(feat.detach().clone())
        else:
            backbone_features_stored = []
        
        # Extract layer features (patch tokens only)
        layer_features, class_tokens = self._extract_layer_features(backbone_features)
        
        # Single unified SAE pass on all layer features
        sae_out = self.sae(layer_features)
        
        # Get reconstructed per-layer features
        layer_reconstructed = sae_out['layer_reconstructed']
        h_sparse = sae_out['h_sparse']
        
        # Prepare features for decoder
        if self.use_sparse_features:
            # Project sparse features to decoder dimension for each layer
            processed_features = []
            for i, proj in enumerate(self.sparse_to_decoder):
                projected = proj(h_sparse)
            # Reconstruct tuple format if needed
                if class_tokens[i] is not None:
                    new_class_token = torch.zeros(
                        class_tokens[i].shape[0], 1, projected.shape[-1],
                        device=class_tokens[i].device, dtype=class_tokens[i].dtype
                    )
                    processed_features.append((projected, new_class_token))
                else:
                    processed_features.append(projected)
        else:
            # Use reconstructed features
            processed_features = []
            for i, recon in enumerate(layer_reconstructed):
                if class_tokens[i] is not None:
                    processed_features.append((recon, class_tokens[i]))
                else:
                    processed_features.append(recon)
        
        # Store intermediate features if needed
        if return_intermediates:
            sparse_features = [h_sparse.detach().clone()]
            reconstructed_features = [r.detach().clone() for r in layer_reconstructed]
        else:
            sparse_features = []
            reconstructed_features = []
        
        # DPT decoder
        logits = self.segmentation_head(processed_features, patch_h, patch_w)
        
        return (
            logits,
            sae_out,  # Return full SAE output dict with all losses
            backbone_features_stored,
            sparse_features,
            reconstructed_features
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], SAEIntermediateFeatures]:
        """Forward pass with mode-dependent output."""
        return_intermediates = (self._output_mode == "debug")
        
        logits, sae_out, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=return_intermediates)
        
        if self._output_mode == "inference":
            return logits
        
        elif self._output_mode == "train":
            # Return comprehensive output dict with all SAE metrics
            return {
                'logits': logits,
                'sae_loss': sae_out['loss_total'],
                'sae_stats': sae_out.get('sparsity_stats'),
                'h_sparse': sae_out['h_sparse'],
                # Individual SAE loss components for debugging
                'loss_reconstruction': sae_out.get('loss_reconstruction'),
                'loss_auxiliary': sae_out.get('loss_auxiliary'),
                'loss_orthogonality': sae_out.get('loss_orthogonality'),
                # For reconstruction quality analysis
                'z_concat': sae_out.get('z_concat'),
                'z_reconstructed': sae_out.get('z_reconstructed'),
            }
        
        else:  # debug mode
            intermediates = SAEIntermediateFeatures(
                backbone_features=backbone_feats,
                sparse_features=sparse_feats,
                reconstructed_features=recon_feats,
                logits=logits,
                sae_loss=sae_out['loss_total'],
                sae_stats=sae_out.get('sparsity_stats'),
            )
            self._cached_intermediates = intermediates
            return intermediates
    
    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """JIT-traceable inference forward."""
        logits, _, _, _, _ = self._forward_impl(x, return_intermediates=False)
        return logits
    
    def forward_with_intermediates(self, x: torch.Tensor) -> SAEIntermediateFeatures:
        """Forward pass returning all intermediate features."""
        logits, sae_out, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=True)
        
        return SAEIntermediateFeatures(
            backbone_features=backbone_feats,
            sparse_features=sparse_feats,
            reconstructed_features=recon_feats,
            logits=logits,
            sae_loss=sae_out['loss_total'],
            sae_stats=sae_out.get('sparsity_stats'),
        )
    
    def get_sparse_features(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get sparse features (unified across all layers).
        
        Args:
            x: Optional input images. If None, uses cached features.
        
        Returns:
            Sparse features tensor (B, P, d_hidden)
        """
        if x is not None:
            intermediates = self.forward_with_intermediates(x)
            return intermediates.sparse_features[0]  # Single unified sparse
        elif self._cached_intermediates is not None:
            return self._cached_intermediates.sparse_features[0]
        else:
            raise RuntimeError(
                "No cached features. Either pass input x or run forward in debug mode first."
            )

    # =========================================================================
    # STEERING API
    # =========================================================================
    
    def steer(
        self,
        x: torch.Tensor,
        interventions: Dict[int, Tuple[str, float]]
    ) -> torch.Tensor:
        """
        Forward pass with steering interventions.
        
        Args:
            x: Input images (B, 3, H, W)
            interventions: {feature_idx: (operation, value)}
                          operations: 'boost', 'suppress', 'clamp'
        
        Returns:
            Steered logits (B, num_classes, H, W)
        
        Example:
            >>> interventions = {
            ...     100: ('boost', 5.0),    # Boost feature 100
            ...     200: ('suppress', 0),   # Suppress feature 200
            ... }
            >>> steered_logits = model.steer(images, interventions)
        """
        B, C, H, W = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # Extract backbone features
        with torch.no_grad():
            backbone_features = self.backbone(x)
        
        # Extract layer features
        layer_features, class_tokens = self._extract_layer_features(backbone_features)
        
        # Apply error-corrected steering or direct steering
        if self.use_error_correction:
            steered_layers = self.sae.error_corrected_steering(layer_features, interventions)
        else:
            h_sparse, _ = self.sae.encode(layer_features)
            h_steered = self.sae.steer_features(h_sparse, interventions)
            z_steered = self.sae.decode(h_steered)
            steered_layers = self.sae.split_layers(z_steered)
        
        # Prepare features for decoder
        processed_features = []
        for i, steered in enumerate(steered_layers):
            if class_tokens[i] is not None:
                processed_features.append((steered, class_tokens[i]))
            else:
                processed_features.append(steered)
        
        # DPT decoder
        logits = self.segmentation_head(processed_features, patch_h, patch_w)
        
        return logits
    
    def find_active_features(
        self,
        x: torch.Tensor,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find the most active features for an input.
        
        Args:
            x: Input images (B, 3, H, W)
            top_k: Number of top features to return
        
        Returns:
            List of (feature_idx, mean_activation) tuples
        """
        with torch.no_grad():
            backbone_features = self.backbone(x)
        
        layer_features, _ = self._extract_layer_features(backbone_features)
        return self.sae.find_active_features(layer_features, top_k=top_k)
    
    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Get the dictionary direction for a feature."""
        return self.sae.get_feature_direction(feature_idx)
    
    def get_feature_layer_direction(
        self,
        feature_idx: int,
        layer_idx: int
    ) -> torch.Tensor:
        """Get the dictionary direction for a feature at a specific layer."""
        return self.sae.get_feature_layer_direction(feature_idx, layer_idx)


class Dinov3TopKSAEUNetSegmentation(SAEModelMixin, nn.Module):
    """
    DINOv3 + TopK SAE with UNet-style decoder.
    
    Uses unified dictionary across all backbone layers.
    
    Args:
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type
        dataset: Dataset type for backbone
        sae_hidden_dim: SAE hidden dimension / dictionary size
        sae_k: Number of active features per token
        n_backbone_layers: Number of backbone layers
        use_sparse_features: Use sparse or reconstructed features
        freeze_backbone: Freeze backbone weights
        load_pretrained_backbone: Whether to load pretrained backbone
        decoder_channels: UNet decoder channel dimensions
        use_error_correction: Use error-corrected steering
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        sae_hidden_dim: int = 16384,
        sae_k: int = 64,
        n_backbone_layers: int = 4,
        use_sparse_features: bool = False,
        freeze_backbone: bool = True,
        load_pretrained_backbone: bool = True,
        decoder_channels: List[int] = [512, 256, 128, 64],
        use_error_correction: bool = True,
        # SAE loss coefficients
        sae_aux_loss_coeff: float = 0.1,  # Coefficient for auxiliary (dead neuron) loss
        # Orthogonality parameters (OrtSAE)
        sae_ortho_coeff: float = 0.0,
        sae_ortho_method: str = "sampled",
        sae_ortho_sample_size: int = 1024,
        sae_ortho_compute_freq: int = 1,
    ):
        super().__init__()
        self._init_sae_mixin()
        
        self.num_classes = num_classes
        self.use_sparse_features = use_sparse_features
        self.freeze_backbone = freeze_backbone
        self.sae_hidden_dim = sae_hidden_dim
        self.sae_k = sae_k
        self.n_backbone_layers = n_backbone_layers
        self.use_error_correction = use_error_correction
        
        # Embedding dimensions
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        self.d_input = embedding_dims[backbone_type]
        
        # Backbone
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=True,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone
        )
        
        # Unified Multi-Layer TopK SAE with OrtSAE support
        self.sae = TopKSparseAutoencoder(
            d_input=self.d_input,
            d_hidden=sae_hidden_dim,
            k=sae_k,
            n_layers=n_backbone_layers,
            aux_loss_coeff=sae_aux_loss_coeff,
            ortho_coeff=sae_ortho_coeff,
            ortho_method=sae_ortho_method,
            ortho_sample_size=sae_ortho_sample_size,
            ortho_compute_freq=sae_ortho_compute_freq,
        )
        
        # Determine decoder input dimension
        if use_sparse_features:
            decoder_in_channels = sae_hidden_dim
        else:
            decoder_in_channels = self.d_input
        
        # UNet decoder
        self.decoder = Dinov3UNetDecoder(
            in_channels=decoder_in_channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
        )
        
        # Projection layer for sparse features (if needed)
        if use_sparse_features:
            self.sparse_to_decoder = nn.ModuleList([
                nn.Linear(sae_hidden_dim, decoder_in_channels)
                for _ in range(n_backbone_layers)
            ])
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor, List, List, List]:
        """Internal forward implementation."""
        B, C, H, W = x.shape
        
        # Extract backbone features
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_features = self.backbone(x)
        else:
            backbone_features = self.backbone(x)
        
        # Store backbone features if needed
        if return_intermediates:
            backbone_features_stored = [f.detach().clone() for f in backbone_features]
        else:
            backbone_features_stored = []
        
        # Convert spatial features to sequence format for SAE
        layer_features = []
        spatial_shapes = []
        for feat in backbone_features:
            B_f, C_f, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B_f, h * w, C_f)
            layer_features.append(feat_flat)
        
        # Single unified SAE pass
        sae_out = self.sae(layer_features)
        
        # Get reconstructed features
        layer_reconstructed = sae_out['layer_reconstructed']
        h_sparse = sae_out['h_sparse']
        
        # Convert back to spatial format
        if self.use_sparse_features:
            processed_features = []
            for i, (proj, (h, w)) in enumerate(zip(self.sparse_to_decoder, spatial_shapes)):
                projected = proj(h_sparse)
                processed = projected.reshape(B, h, w, -1).permute(0, 3, 1, 2)
                processed_features.append(processed)
        else:
            processed_features = []
            for recon, (h, w) in zip(layer_reconstructed, spatial_shapes):
                processed = recon.reshape(B, h, w, -1).permute(0, 3, 1, 2)
                processed_features.append(processed)
        
        # Store intermediate features if needed
        if return_intermediates:
            sparse_features = [h_sparse.detach().clone()]
            reconstructed_features = [r.detach().clone() for r in layer_reconstructed]
        else:
            sparse_features = []
            reconstructed_features = []
        
        # UNet decoder
        logits = self.decoder(processed_features, x.shape)
        
        return (
            logits,
            sae_out,  # Return full SAE output dict with all losses
            backbone_features_stored,
            sparse_features,
            reconstructed_features
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], SAEIntermediateFeatures]:
        """Forward pass with mode-dependent output."""
        return_intermediates = (self._output_mode == "debug")
        
        logits, sae_out, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=return_intermediates)
        
        if self._output_mode == "inference":
            return logits
        
        elif self._output_mode == "train":
            # Return comprehensive output dict with all SAE metrics
            return {
                'logits': logits,
                'sae_loss': sae_out['loss_total'],
                'sae_stats': sae_out.get('sparsity_stats'),
                'h_sparse': sae_out['h_sparse'],
                # Individual SAE loss components for debugging
                'loss_reconstruction': sae_out.get('loss_reconstruction'),
                'loss_auxiliary': sae_out.get('loss_auxiliary'),
                'loss_orthogonality': sae_out.get('loss_orthogonality'),
                # For reconstruction quality analysis
                'z_concat': sae_out.get('z_concat'),
                'z_reconstructed': sae_out.get('z_reconstructed'),
            }
        
        else:  # debug mode
            intermediates = SAEIntermediateFeatures(
                backbone_features=backbone_feats,
                sparse_features=sparse_feats,
                reconstructed_features=recon_feats,
                logits=logits,
                sae_loss=sae_out['loss_total'],
                sae_stats=sae_out.get('sparsity_stats'),
            )
            self._cached_intermediates = intermediates
            return intermediates
    
    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """JIT-traceable inference forward."""
        logits, _, _, _, _ = self._forward_impl(x, return_intermediates=False)
        return logits
    
    def forward_with_intermediates(self, x: torch.Tensor) -> SAEIntermediateFeatures:
        """Forward pass returning all intermediate features."""
        logits, sae_out, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=True)
        
        return SAEIntermediateFeatures(
            backbone_features=backbone_feats,
            sparse_features=sparse_feats,
            reconstructed_features=recon_feats,
            logits=logits,
            sae_loss=sae_out['loss_total'],
            sae_stats=sae_out.get('sparsity_stats'),
        )
    
    def get_sparse_features(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get sparse features (unified across all layers)."""
        if x is not None:
            intermediates = self.forward_with_intermediates(x)
            return intermediates.sparse_features[0]
        elif self._cached_intermediates is not None:
            return self._cached_intermediates.sparse_features[0]
        else:
            raise RuntimeError(
                "No cached features. Either pass input x or run forward in debug mode first."
            )
    
    def steer(
        self,
        x: torch.Tensor,
        interventions: Dict[int, Tuple[str, float]]
    ) -> torch.Tensor:
        """Forward pass with steering interventions."""
        B, C, H, W = x.shape
        
        # Extract backbone features
        with torch.no_grad():
            backbone_features = self.backbone(x)
        
        # Convert to sequence format
        layer_features = []
        spatial_shapes = []
        for feat in backbone_features:
            B_f, C_f, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B_f, h * w, C_f)
            layer_features.append(feat_flat)
        
        # Apply steering
        if self.use_error_correction:
            steered_layers = self.sae.error_corrected_steering(layer_features, interventions)
        else:
            h_sparse, _ = self.sae.encode(layer_features)
            h_steered = self.sae.steer_features(h_sparse, interventions)
            z_steered = self.sae.decode(h_steered)
            steered_layers = self.sae.split_layers(z_steered)
        
        # Convert back to spatial format
        processed_features = []
        for steered, (h, w) in zip(steered_layers, spatial_shapes):
            processed = steered.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            processed_features.append(processed)
        
        # UNet decoder
        logits = self.decoder(processed_features, x.shape)
        
        return logits


def create_topk_sae_segmentation_model(
    model_type: str = "dinov3_topk_sae_dpt",
    num_classes: int = 2,
    backbone_type: str = "vitl",
    dataset: str = "lvd1689m",
    sae_config: Optional[Dict] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create TopK SAE segmentation models.
    
    Args:
        model_type: Type of model ('dinov3_topk_sae_dpt', 'dinov3_topk_sae_unet')
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type
        dataset: Dataset type for backbone
        sae_config: SAE configuration dict with keys:
            - d_hidden: Hidden dimension / dictionary size (default 16384)
            - k: Number of active features (default 64)
            - n_layers: Number of backbone layers (default 4)
            - use_sparse_features: Use sparse or reconstructed features (default False)
            - use_error_correction: Use error-corrected steering (default True)
            - aux_loss_coeff: Auxiliary loss coefficient (default 0.1)
            - ortho_coeff: Orthogonality loss coefficient (default 0.0)
            - ortho_method: Method for computing ortho loss (default "sampled")
            - ortho_sample_size: Sample size for sampled method (default 1024)
            - ortho_compute_freq: Compute frequency (default 1)
        **kwargs: Additional model arguments
    
    Returns:
        Instantiated model
    """
    sae_config = sae_config or {}
    sae_hidden_dim = sae_config.get('d_hidden', 16384)
    sae_k = sae_config.get('k', 64)
    n_backbone_layers = sae_config.get('n_layers', 4)
    use_sparse_features = sae_config.get('use_sparse_features', False)
    use_error_correction = sae_config.get('use_error_correction', True)
    
    # SAE loss coefficients
    sae_aux_loss_coeff = sae_config.get('aux_loss_coeff', 0.1)
    
    # OrtSAE parameters
    sae_ortho_coeff = sae_config.get('ortho_coeff', 0.0)
    sae_ortho_method = sae_config.get('ortho_method', 'sampled')
    sae_ortho_sample_size = sae_config.get('ortho_sample_size', 1024)
    sae_ortho_compute_freq = sae_config.get('ortho_compute_freq', 1)
    
    model_classes = {
        'dinov3_topk_sae_dpt': Dinov3TopKSAEDPTSegmentation,
        'dinov3_topk_sae_unet': Dinov3TopKSAEUNetSegmentation,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(model_classes.keys())}")
    
    return model_classes[model_type](
        num_classes=num_classes,
        backbone_type=backbone_type,
        dataset=dataset,
        sae_hidden_dim=sae_hidden_dim,
        sae_k=sae_k,
        n_backbone_layers=n_backbone_layers,
        use_sparse_features=use_sparse_features,
        use_error_correction=use_error_correction,
        sae_aux_loss_coeff=sae_aux_loss_coeff,
        sae_ortho_coeff=sae_ortho_coeff,
        sae_ortho_method=sae_ortho_method,
        sae_ortho_sample_size=sae_ortho_sample_size,
        sae_ortho_compute_freq=sae_ortho_compute_freq,
        **kwargs
    )
