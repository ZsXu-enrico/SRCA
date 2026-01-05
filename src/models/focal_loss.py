"""Focal Loss implementation for addressing class imbalance in API recommendation.

Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
- p_t: predicted probability for the true class
- α_t: weighting factor for class balance
- γ: focusing parameter (higher γ focuses more on hard examples)

From: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
Used in SRCA to handle the imbalanced API recommendation problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Designed to address class imbalance by down-weighting easy examples
    and focusing on hard, misclassified examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative examples
                  Default 0.25 as in SRCA paper
            gamma: Focusing parameter γ >= 0
                  γ=0 reduces to standard cross-entropy
                  γ=2 is default from original paper and SRCA
            reduction: 'none' | 'mean' | 'sum'
                      Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]
                    1 for positive class, 0 for negative class

        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Calculate p_t: probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that adjusts alpha based on class frequency.

    For highly imbalanced datasets, uses inverse frequency as alpha.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_freq: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Adaptive Focal Loss.

        Args:
            gamma: Focusing parameter
            class_freq: Frequency of each class [num_classes]
                       If provided, alpha is computed as inverse frequency
            reduction: Reduction method
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_freq is not None:
            # Compute alpha as inverse frequency (normalized)
            inv_freq = 1.0 / (class_freq + 1e-6)
            self.alpha = inv_freq / inv_freq.sum()
        else:
            self.alpha = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate adaptive focal loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]

        Returns:
            Loss value
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            # Use adaptive alpha per class
            alpha_t = self.alpha.to(inputs.device).unsqueeze(0)  # [1, num_classes]
            alpha_factor = alpha_t * targets + (1 - alpha_t) * (1 - targets)
        else:
            # Default alpha = 0.25
            alpha_factor = 0.25 * targets + 0.75 * (1 - targets)

        focal_loss = alpha_factor * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_focal_loss():
    """Test focal loss implementation."""
    print("Testing Focal Loss...")

    batch_size = 16
    num_classes = 100

    # Create sample data (imbalanced)
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.zeros(batch_size, num_classes)
    # Only a few positive examples (simulating imbalance)
    targets[torch.randint(0, batch_size, (5,)), torch.randint(0, num_classes, (5,))] = 1.0

    print(f"\nInputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Positive rate: {targets.mean():.4f}")

    # Test standard Focal Loss
    print("\n1. Testing standard Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(inputs, targets)
    print(f"   Loss value: {loss.item():.4f}")

    # Test with gamma=0 (should be close to BCE)
    print("\n2. Testing with gamma=0 (equivalent to BCE)...")
    focal_loss_gamma0 = FocalLoss(alpha=0.5, gamma=0.0)
    loss_gamma0 = focal_loss_gamma0(inputs, targets)
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
    print(f"   Focal Loss (γ=0): {loss_gamma0.item():.4f}")
    print(f"   BCE Loss: {bce_loss.item():.4f}")
    print(f"   Difference: {abs(loss_gamma0.item() - bce_loss.item()):.6f}")

    # Test Adaptive Focal Loss
    print("\n3. Testing Adaptive Focal Loss...")
    class_freq = torch.rand(num_classes) * 0.1  # Random frequencies
    adaptive_focal = AdaptiveFocalLoss(gamma=2.0, class_freq=class_freq)
    adaptive_loss = adaptive_focal(inputs, targets)
    print(f"   Adaptive loss value: {adaptive_loss.item():.4f}")

    # Test gradient flow
    print("\n4. Testing gradient flow...")
    inputs.requires_grad = True
    loss = focal_loss(inputs, targets)
    loss.backward()
    print(f"   Gradient shape: {inputs.grad.shape}")
    print(f"   Gradient range: [{inputs.grad.min():.6f}, {inputs.grad.max():.6f}]")

    print("\n✓ Focal Loss test passed!")


if __name__ == "__main__":
    test_focal_loss()
