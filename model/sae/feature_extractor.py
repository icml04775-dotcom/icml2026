"""
SAE Feature Extractor for visualization and analysis.

This module provides a high-level interface for extracting and visualizing
intermediate features from DINOv3+SAE segmentation models.

Usage:
    from model.sae import SAEFeatureExtractor

    # Load trained model
    model = Dinov3SAEDPTSegmentation(num_classes=9, ...)
    model.load_state_dict(torch.load("checkpoint.pt")["state_dict"])

    # Create extractor
    extractor = SAEFeatureExtractor(model, device="cuda")

    # Extract features for a single image
    features = extractor.extract(image)

    # Access intermediate outputs
    sparse_map = features.get_sparse_feature_map()  # (B, 4096, H, W)
    feature_42 = features.get_feature_activation(42)  # (B, H, W)

    # Visualize
    extractor.visualize_sparse_features(image, class_names, save_path="features.png")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ExtractedFeatures:
    """
    Container for extracted features from SAE model.

    This is a user-friendly wrapper around SAEIntermediateFeatures with
    additional convenience methods for visualization and analysis.

    Attributes:
        backbone: Raw backbone features per level
        sparse: SAE sparse features (h) per level
        reconstructed: SAE reconstructed features (z') per level
        logits: Final segmentation logits
        prediction: Argmax prediction from logits
        sae_loss: SAE reconstruction + sparsity loss
        sae_stats: Sparsity statistics dict
        image_size: Original input image size (H, W)
        feature_size: Spatial size of features (h, w)
    """
    backbone: List[torch.Tensor]
    sparse: List[torch.Tensor]
    reconstructed: List[torch.Tensor]
    logits: torch.Tensor
    prediction: torch.Tensor
    sae_loss: torch.Tensor
    sae_stats: Dict[str, Any]
    image_size: Tuple[int, int]
    feature_size: Tuple[int, int]

    @property
    def num_sparse_features(self) -> int:
        """Number of sparse features (e.g., 4096)."""
        return self.sparse[-1].shape[-1]

    @property
    def sparsity(self) -> float:
        """Overall sparsity of sparse features."""
        sparse = self.sparse[-1]
        return (sparse == 0).float().mean().item()

    @property
    def active_features(self) -> torch.Tensor:
        """Indices of features that are active (non-zero) anywhere."""
        sparse = self.sparse[-1]  # (B, N, d_hidden)
        active = (sparse.sum(dim=[0, 1]) > 0)
        return active.nonzero(as_tuple=True)[0]

    def get_sparse_feature_map(self, level: int = -1) -> torch.Tensor:
        """
        Get sparse features reshaped as spatial feature map.

        Args:
            level: Feature level (-1 for last)

        Returns:
            Tensor of shape (B, d_hidden, h, w)
        """
        sparse = self.sparse[level]
        B, N, D = sparse.shape
        h, w = self.feature_size
        return sparse.reshape(B, h, w, D).permute(0, 3, 1, 2)

    def get_feature_activation(self, feature_idx: int, level: int = -1) -> torch.Tensor:
        """
        Get spatial activation map for a specific sparse feature.

        Args:
            feature_idx: Index of the sparse feature (0 to num_sparse_features-1)
            level: Feature level (-1 for last)

        Returns:
            Tensor of shape (B, h, w) with activation values
        """
        sparse = self.sparse[level]
        B, N, D = sparse.shape
        h, w = self.feature_size
        return sparse[:, :, feature_idx].reshape(B, h, w)

    def get_feature_activation_upsampled(
        self,
        feature_idx: int,
        level: int = -1
    ) -> torch.Tensor:
        """
        Get spatial activation map upsampled to original image size.

        Args:
            feature_idx: Index of the sparse feature
            level: Feature level (-1 for last)

        Returns:
            Tensor of shape (B, H, W) at original image resolution
        """
        activation = self.get_feature_activation(feature_idx, level)
        upsampled = F.interpolate(
            activation.unsqueeze(1).float(),
            size=self.image_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)
        return upsampled

    def get_top_features_for_class(
        self,
        class_idx: int,
        mask: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k sparse features that activate most for a given class.

        Args:
            class_idx: Class index to analyze
            mask: Optional ground truth mask (H, W). If None, uses prediction.
            top_k: Number of top features to return

        Returns:
            Tuple of (feature_indices, mean_activations)
        """
        # Use prediction mask if ground truth not provided
        if mask is None:
            mask = self.prediction.squeeze(0)

        # Get sparse features at last level
        sparse = self.sparse[-1].squeeze(0)  # (N, d_hidden)
        h, w = self.feature_size
        sparse_spatial = sparse.reshape(h, w, -1)  # (h, w, d_hidden)

        # Upsample to mask size
        sparse_up = F.interpolate(
            sparse_spatial.permute(2, 0, 1).unsqueeze(0),
            size=mask.shape,
            mode='bilinear',
            align_corners=True
        ).squeeze(0).permute(1, 2, 0)  # (H, W, d_hidden)

        # Get mean activation per feature for this class
        class_mask = (mask == class_idx)
        if class_mask.any():
            class_activations = sparse_up[class_mask].mean(dim=0)  # (d_hidden,)
            top_vals, top_idx = class_activations.topk(top_k)
            return top_idx, top_vals
        else:
            return torch.tensor([]), torch.tensor([])

    def get_class_feature_association(
        self,
        num_classes: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean activation of each feature for each class.

        Args:
            num_classes: Number of segmentation classes
            mask: Optional ground truth mask (H, W). If None, uses prediction.

        Returns:
            Association matrix of shape (num_features, num_classes)
        """
        if mask is None:
            mask = self.prediction.squeeze(0)

        sparse = self.sparse[-1].squeeze(0)  # (N, d_hidden)
        h, w = self.feature_size
        sparse_spatial = sparse.reshape(h, w, -1)

        sparse_up = F.interpolate(
            sparse_spatial.permute(2, 0, 1).unsqueeze(0),
            size=mask.shape,
            mode='bilinear',
            align_corners=True
        ).squeeze(0).permute(1, 2, 0)

        num_features = sparse_up.shape[-1]
        association = torch.zeros(num_features, num_classes)

        for cls_idx in range(num_classes):
            cls_mask = (mask == cls_idx)
            if cls_mask.any():
                association[:, cls_idx] = sparse_up[cls_mask].mean(dim=0).cpu()

        return association


class SAEFeatureExtractor:
    """
    High-level interface for extracting and visualizing SAE features.

    This class wraps a DINOv3+SAE segmentation model and provides convenient
    methods for feature extraction, visualization, and analysis.

    Args:
        model: DINOv3+SAE segmentation model (Dinov3SAESegmentation,
               Dinov3SAEDPTSegmentation, or Dinov3SAEUNetSegmentation)
        device: Device for computation ('cuda' or 'cpu')

    Example:
        >>> model = Dinov3SAEDPTSegmentation(num_classes=9, ...)
        >>> model.load_state_dict(checkpoint['state_dict'])
        >>>
        >>> extractor = SAEFeatureExtractor(model, device='cuda')
        >>>
        >>> # Extract features
        >>> features = extractor.extract(image)
        >>> print(f"Sparsity: {features.sparsity:.2%}")
        >>> print(f"Active features: {len(features.active_features)}")
        >>>
        >>> # Visualize
        >>> extractor.visualize_sparse_features(image, class_names)
        >>> extractor.visualize_top_features_per_class(dataloader, class_names)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Verify model has required methods
        if not hasattr(model, 'forward_with_intermediates'):
            raise ValueError(
                "Model must have 'forward_with_intermediates' method. "
                "Use Dinov3SAESegmentation, Dinov3SAEDPTSegmentation, or "
                "Dinov3SAEUNetSegmentation."
            )

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> ExtractedFeatures:
        """
        Extract all intermediate features for an image.

        Args:
            image: Input image tensor. Can be:
                   - (3, H, W) single image
                   - (1, 3, H, W) batched single image
                   - (B, 3, H, W) batch of images

        Returns:
            ExtractedFeatures with all intermediate outputs
        """
        self.model.eval()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        B, C, H, W = image.shape

        # Extract using model's debug mode
        intermediates = self.model.forward_with_intermediates(image)

        # Determine feature spatial size
        sparse = intermediates.sparse_features[-1]
        N = sparse.shape[1]
        h = w = int(np.sqrt(N))

        # Get prediction
        prediction = intermediates.logits.argmax(dim=1)

        return ExtractedFeatures(
            backbone=intermediates.backbone_features,
            sparse=intermediates.sparse_features,
            reconstructed=intermediates.reconstructed_features,
            logits=intermediates.logits,
            prediction=prediction,
            sae_loss=intermediates.sae_loss,
            sae_stats=intermediates.sae_stats,
            image_size=(H, W),
            feature_size=(h, w),
        )

    @torch.no_grad()
    def extract_batch(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        show_progress: bool = True
    ) -> List[ExtractedFeatures]:
        """
        Extract features from multiple batches.

        Args:
            dataloader: DataLoader providing images (and optionally masks)
            max_batches: Maximum number of batches to process
            show_progress: Whether to show progress bar

        Returns:
            List of ExtractedFeatures, one per batch
        """
        self.model.eval()
        results = []

        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader

        for batch_idx, batch in enumerate(iterator):
            if max_batches and batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            features = self.extract(images)
            results.append(features)

        return results

    @torch.no_grad()
    def compute_dataset_statistics(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Compute feature statistics over a dataset.

        Args:
            dataloader: DataLoader providing images
            max_batches: Maximum batches to process
            show_progress: Show progress bar

        Returns:
            Dict with:
                - mean_activation: (num_features,) mean per feature
                - std_activation: (num_features,) std per feature
                - activation_frequency: (num_features,) how often each activates
                - max_activation: (num_features,) max value per feature
                - sparsity: Overall sparsity
                - active_count: Number of non-dead features
                - dead_count: Number of dead features
        """
        self.model.eval()

        # Initialize accumulators
        first_batch = True
        d_hidden = None
        activation_sums = None
        activation_sq_sums = None
        activation_counts = None
        activation_maxs = None
        total_patches = 0

        iterator = tqdm(dataloader, desc="Computing statistics") if show_progress else dataloader

        for batch_idx, batch in enumerate(iterator):
            if max_batches and batch_idx >= max_batches:
                break

            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            features = self.extract(images)
            sparse = features.sparse[-1]  # (B, N, d_hidden)

            # Initialize on first batch
            if first_batch:
                d_hidden = sparse.shape[-1]
                activation_sums = torch.zeros(d_hidden, device=self.device)
                activation_sq_sums = torch.zeros(d_hidden, device=self.device)
                activation_counts = torch.zeros(d_hidden, device=self.device)
                activation_maxs = torch.zeros(d_hidden, device=self.device)
                first_batch = False

            # Flatten batch and spatial dims
            sparse_flat = sparse.reshape(-1, d_hidden)

            # Update statistics
            activation_sums += sparse_flat.sum(dim=0)
            activation_sq_sums += (sparse_flat ** 2).sum(dim=0)
            activation_counts += (sparse_flat > 0).float().sum(dim=0)
            activation_maxs = torch.maximum(activation_maxs, sparse_flat.max(dim=0).values)
            total_patches += sparse_flat.shape[0]

        # Compute final statistics
        mean_activation = activation_sums / total_patches
        var_activation = (activation_sq_sums / total_patches) - mean_activation ** 2
        activation_frequency = activation_counts / total_patches

        return {
            'mean_activation': mean_activation.cpu(),
            'std_activation': torch.sqrt(var_activation.clamp(min=0)).cpu(),
            'activation_frequency': activation_frequency.cpu(),
            'max_activation': activation_maxs.cpu(),
            'sparsity': 1 - activation_frequency.mean().item(),
            'active_count': (activation_counts > 0).sum().item(),
            'dead_count': (activation_counts == 0).sum().item(),
            'total_patches': total_patches,
            'd_hidden': d_hidden,
        }

    def visualize_sparse_features(
        self,
        image: torch.Tensor,
        class_names: List[str],
        ground_truth: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 16),
        rows: int = 8,
        cols: int = 512,
    ):
        """
        Visualize all sparse features for an image, color-coded by class.

        Creates a grid showing which features activate for which classes
        in the given image.

        Args:
            image: Input image (3, H, W) or (1, 3, H, W)
            class_names: List of class names
            ground_truth: Optional ground truth mask (H, W)
            save_path: Path to save the figure
            figsize: Figure size
            rows: Number of rows in feature grid
            cols: Number of columns in feature grid

        Returns:
            matplotlib Figure if matplotlib is available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        # Extract features
        features = self.extract(image)
        num_classes = len(class_names)

        # Use ground truth or prediction for class association
        mask = ground_truth if ground_truth is not None else features.prediction.squeeze(0)
        association = features.get_class_feature_association(num_classes, mask.cpu())

        # Prepare image for display
        if image.dim() == 4:
            image = image.squeeze(0)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(3, 3, figure=fig, height_ratios=[1, 3, 0.5])

        # Top row: Image, GT/Prediction, Sparsity stats
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_np)
        ax_img.set_title('Input Image')
        ax_img.axis('off')

        ax_mask = fig.add_subplot(gs[0, 1])
        ax_mask.imshow(mask.cpu() if torch.is_tensor(mask) else mask,
                       cmap='tab10', vmin=0, vmax=num_classes-1)
        ax_mask.set_title('Ground Truth' if ground_truth is not None else 'Prediction')
        ax_mask.axis('off')

        ax_stats = fig.add_subplot(gs[0, 2])
        stats_text = (
            f"Sparsity: {features.sparsity:.2%}\n"
            f"Active features: {len(features.active_features)}\n"
            f"Feature size: {features.feature_size}\n"
            f"SAE Loss: {features.sae_loss.item():.4f}"
        )
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                      fontsize=12, verticalalignment='center', fontfamily='monospace')
        ax_stats.axis('off')
        ax_stats.set_title('Statistics')

        # Middle: Feature grid
        ax_grid = fig.add_subplot(gs[1, :])

        # Create class colors
        if num_classes <= 10:
            class_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_classes, :3]
        else:
            class_colors = plt.cm.tab20(np.linspace(0, 1, min(num_classes, 20)))[:num_classes, :3]

        # Normalize associations
        assoc_norm = association / (association.max() + 1e-8)
        assoc_norm = assoc_norm.clamp(0, 1)

        num_features = association.shape[0]
        target_size = rows * cols

        # Pad if necessary
        if num_features < target_size:
            padding = torch.zeros(target_size - num_features, num_classes)
            assoc_norm = torch.cat([assoc_norm, padding], dim=0)

        # Create RGB grid
        rgb_image = np.zeros((rows, cols, 3))

        for feature_idx in range(min(num_features, target_size)):
            row = feature_idx // cols
            col = feature_idx % cols

            class_weights = assoc_norm[feature_idx].numpy()
            dominant_class = class_weights.argmax()
            dominant_weight = class_weights[dominant_class]

            if dominant_weight > 0:
                rgb_image[row, col] = class_colors[dominant_class] * dominant_weight

        ax_grid.imshow(rgb_image, aspect='auto', interpolation='nearest')
        ax_grid.set_xlabel('Feature Index (mod 512)')
        ax_grid.set_ylabel('Feature Row')
        ax_grid.set_title('SAE Features Colored by Dominant Class Activation')
        ax_grid.set_yticks(range(rows))
        ax_grid.set_yticklabels([f'{i*cols}-{(i+1)*cols-1}' for i in range(rows)])

        # Bottom: Legend
        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis('off')
        legend_patches = [Patch(facecolor=class_colors[i], label=class_names[i])
                          for i in range(num_classes)]
        ax_legend.legend(handles=legend_patches, loc='center',
                         ncol=min(num_classes, 6), fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def visualize_feature_heatmap(
        self,
        image: torch.Tensor,
        feature_idx: int,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
    ):
        """
        Visualize activation heatmap for a single feature.

        Args:
            image: Input image (3, H, W)
            feature_idx: Index of the feature to visualize
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        features = self.extract(image)
        heatmap = features.get_feature_activation_upsampled(feature_idx)

        # Prepare image
        if image.dim() == 4:
            image = image.squeeze(0)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Heatmap
        hm = heatmap.squeeze().cpu().numpy()
        axes[1].imshow(hm, cmap='hot')
        axes[1].set_title(f'Feature {feature_idx} Activation')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(img_np)
        axes[2].imshow(hm, alpha=0.5, cmap='hot')
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def visualize_top_features_per_class(
        self,
        image: torch.Tensor,
        class_names: List[str],
        ground_truth: Optional[torch.Tensor] = None,
        top_k: int = 5,
        save_path: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ):
        """
        Show top-k features that activate most for each class.

        Args:
            image: Input image (3, H, W)
            class_names: List of class names
            ground_truth: Optional ground truth mask
            top_k: Number of top features to show per class
            save_path: Path to save figure
            figsize: Figure size (auto-calculated if None)

        Returns:
            matplotlib Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        features = self.extract(image)
        num_classes = len(class_names)
        mask = ground_truth if ground_truth is not None else features.prediction.squeeze(0)

        if figsize is None:
            figsize = (4 * top_k, 3 * num_classes)

        fig, axes = plt.subplots(num_classes, top_k + 1, figsize=figsize)

        for cls_idx, cls_name in enumerate(class_names):
            top_idx, top_vals = features.get_top_features_for_class(
                cls_idx, mask.cpu(), top_k=top_k
            )

            # Class label
            axes[cls_idx, 0].text(0.5, 0.5, cls_name, ha='center', va='center',
                                   fontsize=12, fontweight='bold')
            axes[cls_idx, 0].axis('off')

            # Top feature heatmaps
            for k in range(top_k):
                ax = axes[cls_idx, k + 1]

                if k < len(top_idx):
                    feat_idx = top_idx[k].item()
                    feat_val = top_vals[k].item()

                    heatmap = features.get_feature_activation_upsampled(feat_idx)
                    hm = heatmap.squeeze().cpu().numpy()

                    ax.imshow(hm, cmap='hot')
                    ax.set_title(f'F{feat_idx}\n{feat_val:.2f}', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

                ax.axis('off')

        plt.suptitle('Top Features per Class', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def export_feature_report(
        self,
        dataloader,
        output_dir: str,
        class_names: List[str],
        max_images: int = 20,
        show_progress: bool = True,
    ):
        """
        Generate comprehensive HTML report for SAE features.

        Creates:
        - Summary statistics page
        - Per-image feature visualizations
        - Class-feature association matrix
        - Index HTML file

        Args:
            dataloader: DataLoader providing (images, masks)
            output_dir: Output directory for report
            class_names: List of class names
            max_images: Maximum images to include
            show_progress: Show progress bar
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for report generation")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'images').mkdir(exist_ok=True)

        # Compute dataset statistics
        stats = self.compute_dataset_statistics(
            dataloader, max_batches=max_images, show_progress=show_progress
        )

        # Generate per-image visualizations
        iterator = tqdm(dataloader, desc="Generating visualizations") if show_progress else dataloader

        image_files = []
        for batch_idx, batch in enumerate(iterator):
            if batch_idx >= max_images:
                break

            if isinstance(batch, (tuple, list)):
                images = batch[0]
                masks = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                masks = None

            # Process each image in batch
            for i in range(images.shape[0]):
                img_idx = batch_idx * images.shape[0] + i
                if img_idx >= max_images:
                    break

                img = images[i]
                mask = masks[i] if masks is not None else None

                save_name = f'image_{img_idx:04d}.png'
                self.visualize_sparse_features(
                    img, class_names, mask,
                    save_path=str(output_dir / 'images' / save_name)
                )
                plt.close()

                image_files.append(save_name)

        # Generate HTML report
        self._generate_html_report(output_dir, stats, class_names, image_files)

        print(f"Report generated at {output_dir / 'index.html'}")

    def _generate_html_report(
        self,
        output_dir: Path,
        stats: Dict,
        class_names: List[str],
        image_files: List[str]
    ):
        """Generate HTML index page."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SAE Feature Extraction Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .image-card {{ background: #f8f9fa; padding: 10px; border-radius: 6px; }}
        .image-card img {{ max-width: 100%; border-radius: 4px; }}
        .class-list {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .class-badge {{ padding: 5px 12px; border-radius: 20px; background: #e3f2fd; color: #1565c0; }}
    </style>
</head>
<body>
    <h1>SAE Feature Extraction Report</h1>

    <div class="section">
        <h2>Dataset Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['d_hidden']}</div>
                <div class="stat-label">Total Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['active_count']}</div>
                <div class="stat-label">Active Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['dead_count']}</div>
                <div class="stat-label">Dead Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['sparsity']:.1%}</div>
                <div class="stat-label">Overall Sparsity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['total_patches']:,}</div>
                <div class="stat-label">Total Patches</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Classes</h2>
        <div class="class-list">
"""
        for i, name in enumerate(class_names):
            html_content += f'            <span class="class-badge">{i}: {name}</span>\n'

        html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Per-Image Feature Analysis</h2>
        <div class="image-grid">
"""
        for img_file in image_files:
            html_content += f"""
            <div class="image-card">
                <img src="images/{img_file}" alt="{img_file}">
            </div>
"""

        html_content += """
        </div>
    </div>
</body>
</html>
"""
        with open(output_dir / 'index.html', 'w') as f:
            f.write(html_content)
