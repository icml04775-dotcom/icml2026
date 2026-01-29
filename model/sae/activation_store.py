"""
Activation Store for SAE Training.

This module provides utilities for collecting, caching, and serving
backbone activations for efficient SAE training.

Key Features:
- Extract activations from frozen DINOv3 backbone
- Cache activations to disk for repeated training runs
- Shuffle and batch activations for training
- Memory-efficient streaming for large datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Iterator, Optional, Union, List, Tuple
import numpy as np
from tqdm import tqdm


class ActivationStore:
    """
    Store and manage backbone activations for SAE training.

    This class handles:
    - Extracting activations from DINOv3 backbone
    - Caching activations to disk for efficient training
    - Shuffling and batching activations
    - Memory-efficient iteration

    Args:
        backbone: DINOv3 backbone model (or any model returning features)
        cache_dir: Directory to cache activations (optional)
        device: Device for computation

    Example:
        >>> backbone = Dinov3Backbone(weights_path="path/to/weights.pth")
        >>> store = ActivationStore(backbone, cache_dir="./activation_cache")
        >>> store.collect_activations(train_dataloader, max_samples=100000)
        >>> store.save_cache()
        >>>
        >>> # Later, for training:
        >>> store.load_cache()
        >>> for batch in store.get_batches(batch_size=128):
        ...     output = sae(batch)
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        self.backbone = backbone
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.activations: Optional[torch.Tensor] = None
        self.metadata: dict = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def collect_activations(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        layer_idx: int = -1,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Collect activations from backbone for all images in dataloader.

        Args:
            dataloader: DataLoader providing images
            max_samples: Maximum number of patch activations to collect
            layer_idx: Which intermediate layer to extract (-1 for last)
            show_progress: Whether to show progress bar

        Returns:
            Tensor of activations (N_total_patches, d_input)
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not provided. Cannot collect activations.")

        self.backbone.eval()
        self.backbone.to(self.device)
        all_activations = []
        total_patches = 0

        iterator = tqdm(dataloader, desc="Collecting activations") if show_progress else dataloader

        for batch in iterator:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Get backbone features
            features = self.backbone(images)

            # Select layer
            if isinstance(features, (tuple, list)):
                feat = features[layer_idx]
            else:
                feat = features

            # Handle tuple outputs (patch_tokens, class_token) from DINOv3
            if isinstance(feat, tuple):
                feat = feat[0]  # Use patch tokens, ignore class token

            # Handle different feature formats
            if feat.dim() == 4:
                # (B, C, H, W) -> (B*H*W, C)
                B, C, H, W = feat.shape
                feat_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            elif feat.dim() == 3:
                # (B, N, C) -> (B*N, C)
                B, N, C = feat.shape
                feat_flat = feat.reshape(B * N, C)
            else:
                feat_flat = feat

            all_activations.append(feat_flat.cpu())
            total_patches += feat_flat.shape[0]

            if max_samples and total_patches >= max_samples:
                break

        activations = torch.cat(all_activations, dim=0)

        if max_samples:
            activations = activations[:max_samples]

        self.activations = activations
        self.metadata = {
            'num_samples': len(activations),
            'd_input': activations.shape[-1],
            'layer_idx': layer_idx,
        }

        return activations

    def save_cache(self, filename: str = "activations.pt"):
        """
        Save collected activations to cache.

        Args:
            filename: Name of the cache file
        """
        if self.cache_dir is None:
            raise RuntimeError("No cache directory specified")

        if self.activations is None:
            raise RuntimeError("No activations to save. Call collect_activations first.")

        cache_path = self.cache_dir / filename
        torch.save({
            'activations': self.activations,
            'metadata': self.metadata,
        }, cache_path)
        print(f"Saved {len(self.activations)} activations to {cache_path}")

    def load_cache(self, filename: str = "activations.pt") -> Optional[torch.Tensor]:
        """
        Load activations from cache.

        Args:
            filename: Name of the cache file

        Returns:
            Loaded activations tensor, or None if cache doesn't exist
        """
        if self.cache_dir is None:
            raise RuntimeError("No cache directory specified")

        cache_path = self.cache_dir / filename
        if not cache_path.exists():
            print(f"Cache file not found: {cache_path}")
            return None

        data = torch.load(cache_path)
        self.activations = data['activations']
        self.metadata = data.get('metadata', {})
        print(f"Loaded {len(self.activations)} activations from cache")
        return self.activations

    def get_batches(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> Iterator[torch.Tensor]:
        """
        Yield batches of activations.

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle before yielding
            drop_last: Whether to drop the last incomplete batch

        Yields:
            Batches of activations (batch_size, d_input)
        """
        if self.activations is None:
            raise RuntimeError("No activations collected. Call collect_activations or load_cache first.")

        n_samples = len(self.activations)
        indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]

            if drop_last and len(batch_indices) < batch_size:
                break

            yield self.activations[batch_indices].to(self.device)

    def get_dataloader(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Get a PyTorch DataLoader for the activations.

        This is useful for more complex training loops that need
        DataLoader features like multiple workers.

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer

        Returns:
            DataLoader yielding activation batches
        """
        if self.activations is None:
            raise RuntimeError("No activations collected.")

        dataset = TensorDataset(self.activations)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def sample(self, n: int) -> torch.Tensor:
        """
        Get a random sample of n activations.

        Args:
            n: Number of samples to return

        Returns:
            Tensor of shape (n, d_input)
        """
        if self.activations is None:
            raise RuntimeError("No activations collected.")

        indices = torch.randperm(len(self.activations))[:n]
        return self.activations[indices].to(self.device)

    def normalize(self, method: str = "unit_variance") -> None:
        """
        Normalize activations in-place.

        Args:
            method: Normalization method
                - "unit_variance": Scale so variance = 1
                - "unit_norm": Scale so average L2 norm = sqrt(d_input)
                - "mean_center": Subtract mean
        """
        if self.activations is None:
            raise RuntimeError("No activations collected.")

        if method == "unit_variance":
            std = self.activations.std()
            self.activations = self.activations / (std + 1e-8)
        elif method == "unit_norm":
            d_input = self.activations.shape[-1]
            mean_norm = self.activations.norm(dim=-1).mean()
            target_norm = np.sqrt(d_input)
            self.activations = self.activations * (target_norm / (mean_norm + 1e-8))
        elif method == "mean_center":
            mean = self.activations.mean(dim=0)
            self.activations = self.activations - mean
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self.metadata['normalization'] = method

    def compute_statistics(self) -> dict:
        """
        Compute statistics about the stored activations.

        Returns:
            Dictionary with activation statistics
        """
        if self.activations is None:
            raise RuntimeError("No activations collected.")

        acts = self.activations
        return {
            'num_samples': len(acts),
            'd_input': acts.shape[-1],
            'mean': acts.mean().item(),
            'std': acts.std().item(),
            'min': acts.min().item(),
            'max': acts.max().item(),
            'mean_norm': acts.norm(dim=-1).mean().item(),
            'sparsity': (acts == 0).float().mean().item(),
        }

    def __len__(self) -> int:
        if self.activations is None:
            return 0
        return len(self.activations)

    @property
    def d_input(self) -> int:
        """Return the input dimension of stored activations."""
        if self.activations is None:
            return 0
        return self.activations.shape[-1]


class StreamingActivationStore(ActivationStore):
    """
    Memory-efficient activation store that streams from disk.

    For very large datasets where activations don't fit in memory,
    this class stores activations in shards and streams them during training.

    Args:
        cache_dir: Directory to store activation shards
        shard_size: Number of activations per shard
        device: Device for computation
    """

    def __init__(
        self,
        cache_dir: str,
        shard_size: int = 100000,
        device: str = "cuda"
    ):
        super().__init__(backbone=None, cache_dir=cache_dir, device=device)
        self.shard_size = shard_size
        self.shard_paths: List[Path] = []
        self.current_shard_idx = 0
        self.total_samples = 0

    @torch.no_grad()
    def collect_activations(
        self,
        dataloader: DataLoader,
        backbone: nn.Module,
        max_samples: Optional[int] = None,
        layer_idx: int = -1,
        show_progress: bool = True
    ) -> int:
        """
        Collect activations and save to shards.

        Args:
            dataloader: DataLoader providing images
            backbone: Backbone model for feature extraction
            max_samples: Maximum number of activations to collect
            layer_idx: Which layer to extract
            show_progress: Whether to show progress bar

        Returns:
            Total number of activations collected
        """
        backbone.eval()
        backbone.to(self.device)

        current_shard = []
        shard_idx = 0
        total_patches = 0

        iterator = tqdm(dataloader, desc="Collecting activations") if show_progress else dataloader

        for batch in iterator:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)
            features = backbone(images)

            if isinstance(features, (tuple, list)):
                feat = features[layer_idx]
            else:
                feat = features

            if feat.dim() == 4:
                B, C, H, W = feat.shape
                feat_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            else:
                feat_flat = feat.reshape(-1, feat.shape[-1])

            current_shard.append(feat_flat.cpu())
            total_patches += feat_flat.shape[0]

            # Save shard if full
            current_size = sum(s.shape[0] for s in current_shard)
            if current_size >= self.shard_size:
                self._save_shard(torch.cat(current_shard, dim=0)[:self.shard_size], shard_idx)
                remaining = torch.cat(current_shard, dim=0)[self.shard_size:]
                current_shard = [remaining] if len(remaining) > 0 else []
                shard_idx += 1

            if max_samples and total_patches >= max_samples:
                break

        # Save final shard
        if current_shard:
            final_shard = torch.cat(current_shard, dim=0)
            if max_samples:
                remaining = max_samples - (shard_idx * self.shard_size)
                final_shard = final_shard[:remaining]
            self._save_shard(final_shard, shard_idx)

        self.total_samples = min(total_patches, max_samples) if max_samples else total_patches
        self._save_metadata()

        return self.total_samples

    def _save_shard(self, activations: torch.Tensor, shard_idx: int):
        """Save a shard to disk."""
        shard_path = self.cache_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(activations, shard_path)
        self.shard_paths.append(shard_path)

    def _save_metadata(self):
        """Save metadata about shards."""
        metadata = {
            'shard_paths': [str(p) for p in self.shard_paths],
            'shard_size': self.shard_size,
            'total_samples': self.total_samples,
        }
        torch.save(metadata, self.cache_dir / "metadata.pt")

    def load_cache(self, filename: str = "metadata.pt") -> int:
        """Load shard metadata from cache."""
        metadata_path = self.cache_dir / filename
        if not metadata_path.exists():
            raise RuntimeError(f"Metadata not found: {metadata_path}")

        metadata = torch.load(metadata_path)
        self.shard_paths = [Path(p) for p in metadata['shard_paths']]
        self.shard_size = metadata['shard_size']
        self.total_samples = metadata['total_samples']

        return self.total_samples

    def get_batches(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> Iterator[torch.Tensor]:
        """
        Yield batches by streaming from shards.

        Note: Shuffling is done within shards, not globally.
        For global shuffling, use shuffle_shards() first.
        """
        shard_order = torch.randperm(len(self.shard_paths)) if shuffle else torch.arange(len(self.shard_paths))

        for shard_idx in shard_order:
            shard = torch.load(self.shard_paths[shard_idx])

            indices = torch.randperm(len(shard)) if shuffle else torch.arange(len(shard))

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                if drop_last and len(batch_indices) < batch_size:
                    break
                yield shard[batch_indices].to(self.device)

    def __len__(self) -> int:
        return self.total_samples
