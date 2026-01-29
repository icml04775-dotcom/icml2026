"""
Sparse Autoencoder (SAE) module for DINOv3 interpretability.

This package provides tools for decomposing DINOv3 activations into
sparse, interpretable features using Sparse Autoencoders.

Based on Anthropic's research:
- "Towards Monosemanticity" (2023): https://transformer-circuits.pub/2023/monosemantic-features
- "Scaling Monosemanticity" (2024): https://transformer-circuits.pub/2024/scaling-monosemanticity/

Components:
    - SparseAutoencoder: Core SAE module with soft L1 sparsity
    - TopKSparseAutoencoder: SAE with hard TopK sparsity (guaranteed K active features)
    - JumpReLUSparseAutoencoder: SAE with learnable thresholds
    - SAELoss: Loss function computation
    - ActivationStore: Activation caching for training
    - SAETrainer: Training utilities
    - SAEFeatureExtractor: Feature extraction and visualization wrapper
"""

from .sparse_autoencoder import SparseAutoencoder
from .topk_sparse_autoencoder import TopKSparseAutoencoder, JumpReLUSparseAutoencoder
from .sae_loss import SAELoss, compute_reconstruction_loss, compute_sparsity_loss
from .activation_store import ActivationStore
from .sae_training import SAETrainer, train_sae
from .feature_extractor import SAEFeatureExtractor, ExtractedFeatures

__all__ = [
    "SparseAutoencoder",
    "TopKSparseAutoencoder",
    "JumpReLUSparseAutoencoder",
    "SAELoss",
    "compute_reconstruction_loss",
    "compute_sparsity_loss",
    "ActivationStore",
    "SAETrainer",
    "train_sae",
    "SAEFeatureExtractor",
    "ExtractedFeatures",
]
