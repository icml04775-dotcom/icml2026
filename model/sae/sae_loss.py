"""
SAE Loss Functions.

This module provides loss functions for training Sparse Autoencoders:
- Reconstruction Loss (L2/MSE)
- Sparsity Penalty (L1)
- Auxiliary Loss (Dead Neuron Prevention)

Total Loss:
    L_total = L_reconstruction + λ * L_sparsity + α * L_auxiliary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_reconstruction_loss(
    z: torch.Tensor,
    z_reconstructed: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute reconstruction loss (MSE).

    Measures how well the SAE reconstructs the original activations.

    Formula:
        L_reconstruction = (1/D) * ||z - z'||_2^2

    Args:
        z: Original activations, shape (*, d_input)
        z_reconstructed: Reconstructed activations, shape (*, d_input)
        normalize: If True, pre-normalize activations so average squared L2 norm equals D

    Returns:
        Scalar MSE loss
    """
    if normalize:
        # Pre-normalize so average squared L2 norm equals D
        D = z.shape[-1]
        z_norm = z * torch.sqrt(torch.tensor(D, dtype=z.dtype, device=z.device) /
                                 (z.pow(2).mean() + 1e-8))
        z_recon_norm = z_reconstructed * torch.sqrt(torch.tensor(D, dtype=z.dtype, device=z.device) /
                                                     (z_reconstructed.pow(2).mean() + 1e-8))
        return F.mse_loss(z_recon_norm, z_norm)
    else:
        return F.mse_loss(z_reconstructed, z)


def compute_sparsity_loss(
    h: torch.Tensor,
    normalize_by_dim: bool = True
) -> torch.Tensor:
    """
    Compute sparsity penalty (L1).

    Encourages sparse activations in the hidden layer.

    Formula:
        L_sparsity = (1/M) * ||h||_1 = (1/M) * sum(|h_i|)

    Args:
        h: Sparse hidden activations, shape (*, d_hidden)
        normalize_by_dim: If True, normalize by hidden dimension

    Returns:
        Scalar L1 sparsity loss
    """
    if normalize_by_dim:
        return h.abs().mean()
    else:
        return h.abs().sum(dim=-1).mean()


def compute_auxiliary_loss(
    h: torch.Tensor,
    z: torch.Tensor,
    W_enc: torch.Tensor,
    activation_freq: torch.Tensor,
    dead_threshold: float = 1e-5,
    target_activation: float = 0.01
) -> torch.Tensor:
    """
    Compute auxiliary loss for dead neuron prevention.

    This loss encourages dead neurons to start activating by
    penalizing their distance from a target activation level.

    Formula:
        L_aux = α * mean((W_enc[dead] · z - target)^2)

    Args:
        h: Sparse hidden activations, shape (batch, d_hidden)
        z: Input activations, shape (batch, d_input)
        W_enc: Encoder weight matrix, shape (d_hidden, d_input)
        activation_freq: Per-feature activation frequencies, shape (d_hidden,)
        dead_threshold: Frequency threshold below which neurons are "dead"
        target_activation: Target pre-activation value for dead neurons

    Returns:
        Scalar auxiliary loss
    """
    dead_mask = activation_freq < dead_threshold
    num_dead = dead_mask.sum()

    if num_dead == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    # Get encoder weights for dead neurons
    W_enc_dead = W_enc[dead_mask]  # (num_dead, d_input)

    # Compute pre-activations for dead neurons
    pre_activations = F.linear(z, W_enc_dead)  # (batch, num_dead)

    # Loss: push pre-activations towards target
    aux_loss = F.mse_loss(pre_activations,
                          torch.full_like(pre_activations, target_activation))

    return aux_loss


class SAELoss(nn.Module):
    """
    Combined loss function for Sparse Autoencoder training.

    Computes:
        L_total = L_reconstruction + λ * L_sparsity + α * L_auxiliary

    Args:
        l1_coeff: L1 sparsity penalty coefficient (λ)
        aux_coeff: Auxiliary loss coefficient (α) for dead neuron prevention
        normalize_reconstruction: Whether to normalize activations for reconstruction loss
        dead_threshold: Activation frequency threshold for dead neurons
        use_auxiliary_loss: Whether to compute auxiliary loss for dead neurons

    Example:
        >>> loss_fn = SAELoss(l1_coeff=1e-4, aux_coeff=1e-3)
        >>> loss_dict = loss_fn(z, z_reconstructed, h, W_enc, activation_freq)
        >>> total_loss = loss_dict['total']
    """

    def __init__(
        self,
        l1_coeff: float = 1e-4,
        aux_coeff: float = 1e-3,
        normalize_reconstruction: bool = False,
        dead_threshold: float = 1e-5,
        use_auxiliary_loss: bool = True,
    ):
        super().__init__()
        self.l1_coeff = l1_coeff
        self.aux_coeff = aux_coeff
        self.normalize_reconstruction = normalize_reconstruction
        self.dead_threshold = dead_threshold
        self.use_auxiliary_loss = use_auxiliary_loss

    def forward(
        self,
        z: torch.Tensor,
        z_reconstructed: torch.Tensor,
        h: torch.Tensor,
        W_enc: Optional[torch.Tensor] = None,
        activation_freq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            z: Original activations
            z_reconstructed: Reconstructed activations
            h: Sparse hidden activations
            W_enc: Encoder weights (required if use_auxiliary_loss=True)
            activation_freq: Per-feature activation frequencies

        Returns:
            Dictionary with loss components:
                - reconstruction: MSE reconstruction loss
                - sparsity: L1 sparsity penalty
                - auxiliary: Auxiliary loss (if enabled)
                - total: Combined total loss
        """
        # Flatten if needed
        z_flat = z.reshape(-1, z.shape[-1]) if z.dim() > 2 else z
        z_recon_flat = z_reconstructed.reshape(-1, z_reconstructed.shape[-1]) if z_reconstructed.dim() > 2 else z_reconstructed
        h_flat = h.reshape(-1, h.shape[-1]) if h.dim() > 2 else h

        # Reconstruction loss
        loss_recon = compute_reconstruction_loss(
            z_flat, z_recon_flat,
            normalize=self.normalize_reconstruction
        )

        # Sparsity loss
        loss_sparsity = compute_sparsity_loss(h_flat)

        # Total loss
        total = loss_recon + self.l1_coeff * loss_sparsity

        result = {
            'reconstruction': loss_recon,
            'sparsity': loss_sparsity,
            'total': total,
        }

        # Auxiliary loss for dead neurons
        if self.use_auxiliary_loss and W_enc is not None and activation_freq is not None:
            loss_aux = compute_auxiliary_loss(
                h_flat, z_flat, W_enc, activation_freq,
                dead_threshold=self.dead_threshold
            )
            result['auxiliary'] = loss_aux
            result['total'] = total + self.aux_coeff * loss_aux
        else:
            result['auxiliary'] = torch.tensor(0.0, device=z.device)

        return result

    def extra_repr(self) -> str:
        return f"l1_coeff={self.l1_coeff}, aux_coeff={self.aux_coeff}"


class SparsityScheduler:
    """
    Scheduler for L1 sparsity coefficient.

    Allows gradual increase of sparsity penalty during training,
    which can help the SAE learn good reconstructions first before
    enforcing sparsity.

    Args:
        initial_coeff: Starting L1 coefficient
        final_coeff: Target L1 coefficient
        warmup_steps: Number of steps to linearly increase from initial to final

    Example:
        >>> scheduler = SparsityScheduler(initial_coeff=0, final_coeff=1e-4, warmup_steps=1000)
        >>> for step in range(2000):
        ...     l1_coeff = scheduler.get_coeff(step)
        ...     # Use l1_coeff in loss computation
    """

    def __init__(
        self,
        initial_coeff: float = 0.0,
        final_coeff: float = 1e-4,
        warmup_steps: int = 1000,
    ):
        self.initial_coeff = initial_coeff
        self.final_coeff = final_coeff
        self.warmup_steps = warmup_steps

    def get_coeff(self, step: int) -> float:
        """Get L1 coefficient for the given step."""
        if step >= self.warmup_steps:
            return self.final_coeff

        # Linear warmup
        progress = step / self.warmup_steps
        return self.initial_coeff + progress * (self.final_coeff - self.initial_coeff)
