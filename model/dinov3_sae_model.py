"""
DINOv3 + SAE Integrated Segmentation Models.

This module provides segmentation models that integrate Sparse Autoencoders
with DINOv3 backbones for interpretable segmentation.

Models:
    - Dinov3SAESegmentation: Main integrated model
    - Dinov3SAEDPTSegmentation: SAE with DPT decoder
    - Dinov3SAEUNetSegmentation: SAE with UNet decoder

Output Modes:
    - "inference": Returns only logits tensor (JIT-traceable)
    - "train": Returns dict with logits, sae_loss, sae_stats
    - "debug": Returns dict with all intermediate features for visualization

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    DINOv3 + SAE Pipeline                             │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Input Image (B, 3, H, W)                                           │
    │         ↓                                                            │
    │  ┌──────────────────────┐                                           │
    │  │  DINOv3 Backbone     │  ← Frozen ViT-L (24 blocks)               │
    │  └──────────────────────┘                                           │
    │         ↓                                                            │
    │  Patch Features z: (B, N_patches, D_dino)                           │
    │         ↓                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │              SPARSE AUTOENCODER (per patch)                   │   │
    │  │   z -> Encoder -> ReLU -> sparse h -> Decoder -> z'          │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │         ↓                                                            │
    │  [Option A: Use z' for segmentation] ← Reconstructed features       │
    │  [Option B: Use h for segmentation]  ← Sparse features              │
    │         ↓                                                            │
    │  ┌──────────────────────┐                                           │
    │  │  Segmentation Head   │  (DPT or UNet decoder)                    │
    │  └──────────────────────┘                                           │
    │         ↓                                                            │
    │  Segmentation Output (B, num_classes, H, W)                         │
    └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
from dataclasses import dataclass

from .sae import SparseAutoencoder
from .dinov3_model import (
    Dinov3Backbone,
    Dinov3SegmentationHead,
    Dinov3UNetDecoder,
)
from .depth_dpt import DPTHead


@dataclass
class SAEIntermediateFeatures:
    """Container for intermediate features from SAE model."""
    backbone_features: List[torch.Tensor]      # Raw backbone outputs
    sparse_features: List[torch.Tensor]        # SAE encoder outputs (h)
    reconstructed_features: List[torch.Tensor] # SAE decoder outputs (z')
    logits: torch.Tensor                       # Final segmentation
    sae_loss: torch.Tensor                     # SAE loss
    sae_stats: Dict[str, Any]                  # Sparsity statistics

    def get_sparse_feature_map(self, level: int = -1) -> torch.Tensor:
        """
        Get sparse features as spatial feature map.

        Args:
            level: Feature level (-1 for last)

        Returns:
            Tensor of shape (B, d_hidden, H, W)
        """
        sparse = self.sparse_features[level]
        B, N, D = sparse.shape
        h = w = int(N ** 0.5)
        return sparse.reshape(B, h, w, D).permute(0, 3, 1, 2)

    def get_feature_activation(self, feature_idx: int, level: int = -1) -> torch.Tensor:
        """
        Get activation map for a specific sparse feature.

        Args:
            feature_idx: Index of sparse feature (0-4095)
            level: Feature level (-1 for last)

        Returns:
            Tensor of shape (B, H, W) with activation values
        """
        sparse = self.sparse_features[level]
        B, N, D = sparse.shape
        h = w = int(N ** 0.5)
        return sparse[:, :, feature_idx].reshape(B, h, w)


class SAEModelMixin:
    """
    Mixin providing output mode functionality for SAE models.

    Supports three output modes:
    - "inference": Returns only logits tensor (for deployment)
    - "train": Returns dict with logits, sae_loss, sae_stats (for training)
    - "debug": Returns dict with all intermediate features (for visualization)
    """

    def _init_sae_mixin(self):
        """Initialize mixin state. Call in __init__."""
        self._output_mode = "train"
        self._cached_intermediates: Optional[SAEIntermediateFeatures] = None

    # def train(self, mode: bool = True):
    #     """
    #     Override to sync _output_mode with PyTorch training state.
        
    #     When mode=True (training): _output_mode = "train" (returns dict with sae_loss)
    #     When mode=False (eval): _output_mode = "inference" (returns logits only)
        
    #     Args:
    #         mode: If True, set to training mode. If False, set to eval mode.
            
    #     Returns:
    #         self
    #     """
    #     super().train(mode)
    #     if mode:
    #         self._output_mode = "train"
    #     else:
    #         self._output_mode = "inference"
    #     return self

    # Note: eval() override not needed - nn.Module.eval() calls self.train(False)
    # which uses our overridden train() method above

    @property
    def output_mode(self) -> str:
        return self._output_mode

    def set_output_mode(self, mode: str):
        """
        Set output mode.

        Args:
            mode: One of "inference", "train", "debug"
        """
        assert mode in ("inference", "train", "debug"), \
            f"Invalid mode: {mode}. Must be 'inference', 'train', or 'debug'"
        self._output_mode = mode

    @contextmanager
    def inference_mode(self):
        """Context manager for temporary inference mode."""
        old_mode = self._output_mode
        self._output_mode = "inference"
        try:
            yield
        finally:
            self._output_mode = old_mode

    @contextmanager
    def debug_mode(self):
        """Context manager for temporary debug mode."""
        old_mode = self._output_mode
        self._output_mode = "debug"
        try:
            yield
        finally:
            self._output_mode = old_mode

    def get_cached_intermediates(self) -> Optional[SAEIntermediateFeatures]:
        """Get cached intermediate features from last forward pass."""
        return self._cached_intermediates

    def clear_cache(self):
        """Clear cached intermediate features."""
        self._cached_intermediates = None


class Dinov3SAESegmentation(SAEModelMixin, nn.Module):
    """
    DINOv3 segmentation with Sparse Autoencoder for interpretability.

    This model integrates an SAE into the DINOv3 segmentation pipeline,
    allowing for interpretable feature analysis while maintaining
    segmentation performance.

    Output Modes:
        - "inference": Returns logits tensor only (JIT-traceable)
        - "train": Returns dict with logits, sae_loss, sae_stats
        - "debug": Returns SAEIntermediateFeatures with all intermediate outputs

    Args:
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type ('vitl', 'vitb', 'vitg')
        dataset: Dataset type for backbone ('lvd1689m' or 'sat493m')
        sae_hidden_dim: Hidden dimension for SAE (expansion factor * d_input)
        sae_l1_coeff: L1 sparsity coefficient for SAE
        use_sparse_features: Whether to use sparse features for segmentation
        freeze_backbone: Whether to freeze the backbone during training
        load_pretrained_backbone: Whether to load pretrained backbone weights
        load_pretrained_sae: Path to pretrained SAE checkpoint (optional)
        sae_loss_weight: Weight for SAE loss in joint training

    Example:
        >>> model = Dinov3SAESegmentation(num_classes=9, backbone_type="vitl")
        >>>
        >>> # Training mode (default)
        >>> output = model(images)
        >>> loss = output['sae_loss'] + criterion(output['logits'], labels)
        >>>
        >>> # Inference mode
        >>> model.set_output_mode("inference")
        >>> logits = model(images)  # Returns tensor directly
        >>>
        >>> # Debug mode for visualization
        >>> with model.debug_mode():
        ...     features = model(images)
        ...     sparse_map = features.get_sparse_feature_map()
    """

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        sae_hidden_dim: int = 4096,
        sae_l1_coeff: float = 1e-4,
        use_sparse_features: bool = False,
        freeze_backbone: bool = True,
        load_pretrained_backbone: bool = True,
        load_pretrained_sae: Optional[str] = None,
        sae_loss_weight: float = 0.1,
        decoder_features: int = 256,
    ):
        super().__init__()
        self._init_sae_mixin()

        self.num_classes = num_classes
        self.use_sparse_features = use_sparse_features
        self.freeze_backbone = freeze_backbone
        self.sae_loss_weight = sae_loss_weight
        self.sae_hidden_dim = sae_hidden_dim

        # Embedding dimensions for different backbone types
        embedding_dims = {"vitl": 1024, "vitb": 768, "vitg": 1536}
        self.d_input = embedding_dims[backbone_type]

        # DINOv3 backbone
        self.backbone = Dinov3Backbone(
            backbone_type=backbone_type,
            dataset=dataset,
            reshape=True,
            return_class_token=False,
            load_pretrained_backbone=load_pretrained_backbone
        )

        # Sparse Autoencoder
        self.sae = SparseAutoencoder(
            d_input=self.d_input,
            d_hidden=sae_hidden_dim,
            l1_coeff=sae_l1_coeff,
        )

        # Load pretrained SAE if provided
        if load_pretrained_sae:
            self._load_pretrained_sae(load_pretrained_sae)

        # Determine decoder input dimension
        if use_sparse_features:
            decoder_in_channels = sae_hidden_dim
        else:
            decoder_in_channels = self.d_input

        # Segmentation decoder
        self.decoder = Dinov3SegmentationHead(
            in_channels=decoder_in_channels,
            features=decoder_features,
            num_classes=num_classes,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False
        )

    def _load_pretrained_sae(self, checkpoint_path: str):
        """Load pretrained SAE weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.sae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.sae.load_state_dict(checkpoint)
        print(f"Loaded pretrained SAE from {checkpoint_path}")

    def _forward_impl(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, List, List, List]:
        """
        Internal forward implementation.

        Returns:
            Tuple of (logits, sae_loss, sae_stats, backbone_features,
                     sparse_features, reconstructed_features)
        """
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

        # Apply SAE to each feature level
        processed_features = []
        sparse_features = []
        reconstructed_features = []
        total_sae_loss = 0
        sae_stats = None

        for feat in backbone_features:
            B_f, C_f, h, w = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B_f, h * w, C_f)

            # SAE forward
            sae_out = self.sae(feat_flat)
            total_sae_loss = total_sae_loss + sae_out['loss_total']
            sae_stats = sae_out['sparsity_stats']

            # Store intermediate features
            if return_intermediates:
                sparse_features.append(sae_out['h_sparse'].detach().clone())
                reconstructed_features.append(sae_out['z_reconstructed'].detach().clone())

            # Select features for decoder
            if self.use_sparse_features:
                processed = sae_out['h_sparse']
            else:
                processed = sae_out['z_reconstructed']

            # Reshape back to spatial
            processed = processed.reshape(B_f, h, w, -1).permute(0, 3, 1, 2)
            processed_features.append(processed)

        # Pass through decoder
        logits = self.decoder(processed_features, x.shape)
        avg_sae_loss = total_sae_loss / len(backbone_features)

        return (logits, avg_sae_loss, sae_stats,
                backbone_features_stored, sparse_features, reconstructed_features)

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], SAEIntermediateFeatures]:
        """
        Forward pass with mode-dependent output.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Depends on output_mode:
            - "inference": torch.Tensor (logits only)
            - "train": Dict with logits, sae_loss, sae_stats
            - "debug": SAEIntermediateFeatures with all intermediate outputs
        """
        return_intermediates = (self._output_mode == "debug")

        logits, sae_loss, sae_stats, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=return_intermediates)

        if self._output_mode == "inference":
            return logits

        elif self._output_mode == "train":
            return {
                'logits': logits,
                'sae_loss': sae_loss,
                'sae_stats': sae_stats,
            }

        else:  # debug mode
            intermediates = SAEIntermediateFeatures(
                backbone_features=backbone_feats,
                sparse_features=sparse_feats,
                reconstructed_features=recon_feats,
                logits=logits,
                sae_loss=sae_loss,
                sae_stats=sae_stats,
            )
            self._cached_intermediates = intermediates
            return intermediates

    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        JIT-traceable inference forward.

        Always returns logits tensor regardless of output_mode.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        logits, _, _, _, _, _ = self._forward_impl(x, return_intermediates=False)
        return logits

    def forward_with_intermediates(self, x: torch.Tensor) -> SAEIntermediateFeatures:
        """
        Forward pass returning all intermediate features.

        Useful for visualization and debugging.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            SAEIntermediateFeatures containing all intermediate outputs
        """
        logits, sae_loss, sae_stats, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=True)

        return SAEIntermediateFeatures(
            backbone_features=backbone_feats,
            sparse_features=sparse_feats,
            reconstructed_features=recon_feats,
            logits=logits,
            sae_loss=sae_loss,
            sae_stats=sae_stats,
        )

    def get_sparse_features(self, x: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get sparse features.

        If x is provided, runs forward pass first.
        Otherwise returns cached features from last debug-mode forward.

        Args:
            x: Optional input images

        Returns:
            List of sparse feature tensors
        """
        if x is not None:
            intermediates = self.forward_with_intermediates(x)
            return intermediates.sparse_features
        elif self._cached_intermediates is not None:
            return self._cached_intermediates.sparse_features
        else:
            raise RuntimeError(
                "No cached features. Either pass input x or run forward in debug mode first."
            )

    def freeze_sae(self):
        """Freeze SAE weights (for decoder-only training)."""
        for param in self.sae.parameters():
            param.requires_grad = False

    def unfreeze_sae(self):
        """Unfreeze SAE weights (for joint training)."""
        for param in self.sae.parameters():
            param.requires_grad = True


class Dinov3SAEDPTSegmentation(SAEModelMixin, nn.Module):
    """
    DINOv3 + SAE with DPT (Dense Prediction Transformer) decoder.

    This variant uses the full DPT decoder architecture for segmentation,
    which provides better handling of multi-scale features.

    Output Modes:
        - "inference": Returns logits tensor only
        - "train": Returns dict with logits, sae_loss, sae_stats
        - "debug": Returns SAEIntermediateFeatures with all intermediate outputs

    Args:
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type
        dataset: Dataset type for backbone ('lvd1689m' or 'sat493m')
        sae_hidden_dim: SAE hidden dimension
        sae_l1_coeff: L1 sparsity coefficient
        use_sparse_features: Use sparse or reconstructed features
        freeze_backbone: Freeze backbone weights
        load_pretrained_backbone: Whether to load pretrained backbone
        patch_size: ViT patch size (default 16)
        use_auxiliary: Use auxiliary outputs for deep supervision
    """

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        sae_hidden_dim: int = 4096,
        sae_l1_coeff: float = 1e-4,
        use_sparse_features: bool = False,
        freeze_backbone: bool = True,
        load_pretrained_backbone: bool = True,
        patch_size: int = 16,
        use_auxiliary: bool = False,
    ):
        super().__init__()
        self._init_sae_mixin()

        self.num_classes = num_classes
        self.use_sparse_features = use_sparse_features
        self.freeze_backbone = freeze_backbone
        self.patch_size = patch_size
        self.sae_hidden_dim = sae_hidden_dim

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

        # SAE
        self.sae = SparseAutoencoder(
            d_input=self.d_input,
            d_hidden=sae_hidden_dim,
            l1_coeff=sae_l1_coeff,
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

    def _forward_impl(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, List, List, List]:
        """Internal forward implementation."""
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

        # Apply SAE to each feature level
        processed_features = []
        sparse_features = []
        reconstructed_features = []
        total_sae_loss = 0
        sae_stats = None

        for feat in backbone_features:
            if isinstance(feat, tuple):
                patch_tokens, class_token = feat
            else:
                patch_tokens = feat

            # SAE forward on patch tokens
            sae_out = self.sae(patch_tokens)
            total_sae_loss = total_sae_loss + sae_out['loss_total']
            sae_stats = sae_out['sparsity_stats']

            # Store intermediate features
            if return_intermediates:
                sparse_features.append(sae_out['h_sparse'].detach().clone())
                reconstructed_features.append(sae_out['z_reconstructed'].detach().clone())

            if self.use_sparse_features:
                processed = sae_out['h_sparse']
            else:
                processed = sae_out['z_reconstructed']

            # Reconstruct tuple format if needed
            if isinstance(feat, tuple):
                if self.use_sparse_features:
                    new_class_token = torch.zeros(
                        class_token.shape[0], 1, processed.shape[-1],
                        device=class_token.device, dtype=class_token.dtype
                    )
                    processed_features.append((processed, new_class_token))
                else:
                    processed_features.append((processed, class_token))
            else:
                processed_features.append(processed)

        # DPT decoder
        logits = self.segmentation_head(processed_features, patch_h, patch_w)
        avg_sae_loss = total_sae_loss / len(backbone_features)

        return (logits, avg_sae_loss, sae_stats,
                backbone_features_stored, sparse_features, reconstructed_features)

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], SAEIntermediateFeatures]:
        """Forward pass with mode-dependent output."""
        return_intermediates = (self._output_mode == "debug")

        logits, sae_loss, sae_stats, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=return_intermediates)

        if self._output_mode == "inference":
            return logits

        elif self._output_mode == "train":
            return {
                'logits': logits,
                'sae_loss': sae_loss,
                'sae_stats': sae_stats,
            }

        else:  # debug mode
            intermediates = SAEIntermediateFeatures(
                backbone_features=backbone_feats,
                sparse_features=sparse_feats,
                reconstructed_features=recon_feats,
                logits=logits,
                sae_loss=sae_loss,
                sae_stats=sae_stats,
            )
            self._cached_intermediates = intermediates
            return intermediates

    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """JIT-traceable inference forward."""
        logits, _, _, _, _, _ = self._forward_impl(x, return_intermediates=False)
        return logits

    def forward_with_intermediates(self, x: torch.Tensor) -> SAEIntermediateFeatures:
        """Forward pass returning all intermediate features."""
        logits, sae_loss, sae_stats, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=True)

        return SAEIntermediateFeatures(
            backbone_features=backbone_feats,
            sparse_features=sparse_feats,
            reconstructed_features=recon_feats,
            logits=logits,
            sae_loss=sae_loss,
            sae_stats=sae_stats,
        )

    def get_sparse_features(self, x: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Get sparse features."""
        if x is not None:
            intermediates = self.forward_with_intermediates(x)
            return intermediates.sparse_features
        elif self._cached_intermediates is not None:
            return self._cached_intermediates.sparse_features
        else:
            raise RuntimeError(
                "No cached features. Either pass input x or run forward in debug mode first."
            )


class Dinov3SAEUNetSegmentation(SAEModelMixin, nn.Module):
    """
    DINOv3 + SAE with UNet-style decoder.

    Args:
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type
        dataset: Dataset type for backbone
        sae_hidden_dim: SAE hidden dimension
        sae_l1_coeff: L1 sparsity coefficient
        use_sparse_features: Use sparse or reconstructed features
        freeze_backbone: Freeze backbone weights
        load_pretrained_backbone: Whether to load pretrained backbone
        decoder_channels: UNet decoder channel dimensions
    """

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "vitl",
        dataset: str = "lvd1689m",
        sae_hidden_dim: int = 4096,
        sae_l1_coeff: float = 1e-4,
        use_sparse_features: bool = False,
        freeze_backbone: bool = True,
        load_pretrained_backbone: bool = True,
        decoder_channels: List[int] = [512, 256, 128, 64],
    ):
        super().__init__()
        self._init_sae_mixin()

        self.num_classes = num_classes
        self.use_sparse_features = use_sparse_features
        self.freeze_backbone = freeze_backbone
        self.sae_hidden_dim = sae_hidden_dim

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

        # SAE
        self.sae = SparseAutoencoder(
            d_input=self.d_input,
            d_hidden=sae_hidden_dim,
            l1_coeff=sae_l1_coeff,
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

    def _forward_impl(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, List, List, List]:
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

        # Apply SAE
        processed_features = []
        sparse_features = []
        reconstructed_features = []
        total_sae_loss = 0
        sae_stats = None

        for feat in backbone_features:
            B_f, C_f, h, w = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B_f, h * w, C_f)

            sae_out = self.sae(feat_flat)
            total_sae_loss = total_sae_loss + sae_out['loss_total']
            sae_stats = sae_out['sparsity_stats']

            # Store intermediate features
            if return_intermediates:
                sparse_features.append(sae_out['h_sparse'].detach().clone())
                reconstructed_features.append(sae_out['z_reconstructed'].detach().clone())

            if self.use_sparse_features:
                processed = sae_out['h_sparse']
            else:
                processed = sae_out['z_reconstructed']

            processed = processed.reshape(B_f, h, w, -1).permute(0, 3, 1, 2)
            processed_features.append(processed)

        # UNet decoder
        logits = self.decoder(processed_features, x.shape)
        avg_sae_loss = total_sae_loss / len(backbone_features)

        return (logits, avg_sae_loss, sae_stats,
                backbone_features_stored, sparse_features, reconstructed_features)

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], SAEIntermediateFeatures]:
        """Forward pass with mode-dependent output."""
        return_intermediates = (self._output_mode == "debug")

        logits, sae_loss, sae_stats, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=return_intermediates)

        if self._output_mode == "inference":
            return logits

        elif self._output_mode == "train":
            return {
                'logits': logits,
                'sae_loss': sae_loss,
                'sae_stats': sae_stats,
            }

        else:  # debug mode
            intermediates = SAEIntermediateFeatures(
                backbone_features=backbone_feats,
                sparse_features=sparse_feats,
                reconstructed_features=recon_feats,
                logits=logits,
                sae_loss=sae_loss,
                sae_stats=sae_stats,
            )
            self._cached_intermediates = intermediates
            return intermediates

    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """JIT-traceable inference forward."""
        logits, _, _, _, _, _ = self._forward_impl(x, return_intermediates=False)
        return logits

    def forward_with_intermediates(self, x: torch.Tensor) -> SAEIntermediateFeatures:
        """Forward pass returning all intermediate features."""
        logits, sae_loss, sae_stats, backbone_feats, sparse_feats, recon_feats = \
            self._forward_impl(x, return_intermediates=True)

        return SAEIntermediateFeatures(
            backbone_features=backbone_feats,
            sparse_features=sparse_feats,
            reconstructed_features=recon_feats,
            logits=logits,
            sae_loss=sae_loss,
            sae_stats=sae_stats,
        )

    def get_sparse_features(self, x: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Get sparse features."""
        if x is not None:
            intermediates = self.forward_with_intermediates(x)
            return intermediates.sparse_features
        elif self._cached_intermediates is not None:
            return self._cached_intermediates.sparse_features
        else:
            raise RuntimeError(
                "No cached features. Either pass input x or run forward in debug mode first."
            )


def create_sae_segmentation_model(
    model_type: str = "dinov3_sae",
    num_classes: int = 2,
    backbone_type: str = "vitl",
    dataset: str = "lvd1689m",
    sae_config: Optional[Dict] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create SAE segmentation models.

    Args:
        model_type: Type of model ('dinov3_sae', 'dinov3_sae_dpt', 'dinov3_sae_unet')
        num_classes: Number of segmentation classes
        backbone_type: DINOv3 backbone type
        dataset: Dataset type for backbone
        sae_config: SAE configuration dict
        **kwargs: Additional model arguments

    Returns:
        Instantiated model
    """
    sae_config = sae_config or {}
    sae_hidden_dim = sae_config.get('d_hidden', 4096)
    sae_l1_coeff = sae_config.get('l1_coeff', 1e-4)
    use_sparse_features = sae_config.get('use_sparse_features', False)

    model_classes = {
        'dinov3_sae': Dinov3SAESegmentation,
        'dinov3_sae_dpt': Dinov3SAEDPTSegmentation,
        'dinov3_sae_unet': Dinov3SAEUNetSegmentation,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(model_classes.keys())}")

    return model_classes[model_type](
        num_classes=num_classes,
        backbone_type=backbone_type,
        dataset=dataset,
        sae_hidden_dim=sae_hidden_dim,
        sae_l1_coeff=sae_l1_coeff,
        use_sparse_features=use_sparse_features,
        **kwargs
    )
