"""
SAE Training Utilities.

This module provides training utilities for Sparse Autoencoders:
- SAETrainer: Complete training loop with logging and checkpointing
- train_sae: Simple training function for quick experiments
- Learning rate schedulers and optimization utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Callable, Any
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from .sparse_autoencoder import SparseAutoencoder
from .sae_loss import SAELoss, SparsityScheduler
from .activation_store import ActivationStore


class SAETrainer:
    """
    Complete training loop for Sparse Autoencoders.

    Features:
    - Automatic decoder weight normalization
    - Dead neuron resampling
    - Learning rate scheduling
    - Checkpointing and logging
    - Sparsity coefficient warmup

    Args:
        sae: SparseAutoencoder model to train
        activation_store: ActivationStore with training activations
        config: Training configuration dictionary

    Example:
        >>> sae = SparseAutoencoder(d_input=1024, d_hidden=4096)
        >>> store = ActivationStore(backbone)
        >>> store.load_cache()
        >>>
        >>> config = {
        ...     'lr': 1e-4,
        ...     'batch_size': 128,
        ...     'epochs': 50,
        ...     'l1_coeff': 1e-4,
        ... }
        >>> trainer = SAETrainer(sae, store, config)
        >>> history = trainer.train()
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        activation_store: ActivationStore,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.sae = sae.to(device)
        self.activation_store = activation_store
        self.config = config
        self.device = device

        # Training config with defaults
        self.lr = config.get('lr', 1e-4)
        self.batch_size = config.get('batch_size', 128)
        self.epochs = config.get('epochs', 50)
        self.l1_coeff = config.get('l1_coeff', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.grad_clip = config.get('grad_clip', 1.0)

        # Dead neuron handling
        self.resample_interval = config.get('resample_interval', 5)
        self.dead_threshold = config.get('dead_threshold', 1e-5)

        # Sparsity warmup
        self.sparsity_warmup_steps = config.get('sparsity_warmup_steps', 0)

        # Logging
        self.log_interval = config.get('log_interval', 100)
        self.save_interval = config.get('save_interval', 5)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints/sae'))

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.sae.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Initialize scheduler if specified
        scheduler_config = config.get('scheduler', None)
        if scheduler_config:
            self.scheduler = self._create_scheduler(scheduler_config)
        else:
            self.scheduler = None

        # Sparsity scheduler for warmup
        if self.sparsity_warmup_steps > 0:
            self.sparsity_scheduler = SparsityScheduler(
                initial_coeff=0.0,
                final_coeff=self.l1_coeff,
                warmup_steps=self.sparsity_warmup_steps
            )
        else:
            self.sparsity_scheduler = None

        # Loss function
        self.loss_fn = SAELoss(
            l1_coeff=self.l1_coeff,
            aux_coeff=config.get('aux_coeff', 0.0),
            dead_threshold=self.dead_threshold,
            use_auxiliary_loss=config.get('use_auxiliary_loss', False),
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            'loss_total': [],
            'loss_reconstruction': [],
            'loss_sparsity': [],
            'num_active_features': [],
            'dead_neurons': [],
            'learning_rate': [],
        }

        self.global_step = 0
        self.best_loss = float('inf')

    def _create_scheduler(self, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler from config."""
        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'constant':
            return optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def train(self) -> Dict[str, List[float]]:
        """
        Run the complete training loop.

        Returns:
            Training history dictionary
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):
            epoch_metrics = self._train_epoch(epoch)

            # Log epoch summary
            print(f"Epoch {epoch+1}/{self.epochs}: "
                  f"loss={epoch_metrics['loss_total']:.4f}, "
                  f"recon={epoch_metrics['loss_reconstruction']:.4f}, "
                  f"sparsity={epoch_metrics['loss_sparsity']:.4f}, "
                  f"active={epoch_metrics['num_active_features']:.1f}")

            # Resample dead neurons
            if (epoch + 1) % self.resample_interval == 0:
                self._resample_dead_neurons()

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, epoch_metrics['loss_total'])

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

        # Save final checkpoint
        self._save_checkpoint(self.epochs - 1, epoch_metrics['loss_total'], is_final=True)

        return self.history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.sae.train()

        epoch_losses = []
        epoch_recon = []
        epoch_sparsity = []
        epoch_active = []

        for batch in tqdm(self.activation_store.get_batches(self.batch_size),
                          desc=f"Epoch {epoch+1}",
                          leave=False):
            metrics = self._train_step(batch)

            epoch_losses.append(metrics['loss_total'])
            epoch_recon.append(metrics['loss_reconstruction'])
            epoch_sparsity.append(metrics['loss_sparsity'])
            epoch_active.append(metrics['num_active_features'])

            self.global_step += 1

        # Compute epoch averages
        avg_metrics = {
            'loss_total': np.mean(epoch_losses),
            'loss_reconstruction': np.mean(epoch_recon),
            'loss_sparsity': np.mean(epoch_sparsity),
            'num_active_features': np.mean(epoch_active),
        }

        # Update history
        for key, value in avg_metrics.items():
            self.history[key].append(value)

        self.history['dead_neurons'].append(
            self.sae.get_dead_neuron_mask(self.dead_threshold).sum().item()
        )
        self.history['learning_rate'].append(
            self.optimizer.param_groups[0]['lr']
        )

        return avg_metrics

    def _train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()

        # Get current L1 coefficient (with optional warmup)
        if self.sparsity_scheduler:
            current_l1 = self.sparsity_scheduler.get_coeff(self.global_step)
            self.sae.l1_coeff = current_l1

        # Forward pass
        output = self.sae(batch)

        # Compute loss
        loss = output['loss_total']

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()

        # Normalize decoder weights
        self.sae.normalize_decoder_weights()

        # Log periodically
        if self.global_step % self.log_interval == 0:
            print(f"  Step {self.global_step}: "
                  f"loss={loss.item():.4f}, "
                  f"active={output['sparsity_stats']['num_active_features'].item():.1f}")

        return {
            'loss_total': loss.item(),
            'loss_reconstruction': output['loss_reconstruction'].item(),
            'loss_sparsity': output['loss_sparsity'].item(),
            'num_active_features': output['sparsity_stats']['num_active_features'].item(),
        }

    def _resample_dead_neurons(self):
        """Resample dead neurons."""
        sample = self.activation_store.sample(1000)
        dead_count = self.sae.resample_dead_neurons(sample, self.dead_threshold)
        if dead_count > 0:
            print(f"  Resampled {dead_count} dead neurons")

    def _save_checkpoint(self, epoch: int, loss: float, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'history': self.history,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'sae_latest.pt')

        # Save best
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(checkpoint, self.checkpoint_dir / 'sae_best.pt')

        # Save epoch checkpoint
        if is_final:
            torch.save(checkpoint, self.checkpoint_dir / 'sae_final.pt')
        else:
            torch.save(checkpoint, self.checkpoint_dir / f'sae_epoch_{epoch+1}.pt')

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.sae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.history = checkpoint.get('history', self.history)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")


def train_sae(
    sae: SparseAutoencoder,
    activation_store: ActivationStore,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    l1_coeff: Optional[float] = None,
    resample_interval: int = 5,
    dead_threshold: float = 1e-5,
    log_interval: int = 100,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Simple training function for SAE.

    This is a convenience function for quick experiments.
    For more control, use SAETrainer directly.

    Args:
        sae: SparseAutoencoder to train
        activation_store: ActivationStore with training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        l1_coeff: L1 sparsity coefficient (uses sae.l1_coeff if not provided)
        resample_interval: Epochs between dead neuron resampling
        dead_threshold: Threshold for dead neuron detection
        log_interval: Steps between logging
        device: Device for training
        checkpoint_dir: Directory for checkpoints (optional)

    Returns:
        Training history dictionary

    Example:
        >>> sae = SparseAutoencoder(d_input=1024, d_hidden=4096, l1_coeff=1e-4)
        >>> store = ActivationStore(backbone)
        >>> store.collect_activations(dataloader)
        >>> history = train_sae(sae, store, epochs=50)
    """
    if l1_coeff is not None:
        sae.l1_coeff = l1_coeff

    sae = sae.to(device)
    sae.train()

    optimizer = optim.Adam(sae.parameters(), lr=lr)

    history = {
        'loss_total': [],
        'loss_reconstruction': [],
        'loss_sparsity': [],
        'num_active_features': [],
    }

    global_step = 0

    for epoch in range(epochs):
        epoch_losses = []

        for batch in activation_store.get_batches(batch_size=batch_size, shuffle=True):
            output = sae(batch)

            loss = output['loss_total']
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights
            sae.normalize_decoder_weights()

            epoch_losses.append(loss.item())

            if global_step % log_interval == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}, "
                      f"recon={output['loss_reconstruction'].item():.4f}, "
                      f"sparsity={output['loss_sparsity'].item():.4f}, "
                      f"active={output['sparsity_stats']['num_active_features'].item():.1f}")

            global_step += 1

        # Record epoch stats
        history['loss_total'].append(np.mean(epoch_losses))
        history['loss_reconstruction'].append(output['loss_reconstruction'].item())
        history['loss_sparsity'].append(output['loss_sparsity'].item())
        history['num_active_features'].append(
            output['sparsity_stats']['num_active_features'].item()
        )

        # Resample dead neurons
        if (epoch + 1) % resample_interval == 0:
            sample = activation_store.sample(1000)
            dead_count = sae.resample_dead_neurons(sample, threshold=dead_threshold)
            if dead_count > 0:
                print(f"Epoch {epoch+1}: Resampled {dead_count} dead neurons")

        print(f"Epoch {epoch+1}/{epochs}: avg_loss={np.mean(epoch_losses):.4f}")

    # Save final model if checkpoint_dir specified
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': sae.state_dict(),
            'history': history,
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'l1_coeff': sae.l1_coeff,
            }
        }, checkpoint_path / 'sae_final.pt')

    return history
