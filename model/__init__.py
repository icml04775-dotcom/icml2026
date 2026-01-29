"""
Model package for DINOv3 + SAE segmentation.

This package contains:
- DINOv3 backbone implementation
- TopK Sparse Autoencoder (SAE)
- DPT decoder for dense prediction
- Segmentation model combining all components
"""

from .dinov3_sae_topk_model import (
    Dinov3TopKSAEDPTSegmentation,
    Dinov3TopKSAEUNetSegmentation,
)

__all__ = [
    "Dinov3TopKSAEDPTSegmentation",
    "Dinov3TopKSAEUNetSegmentation",
]
