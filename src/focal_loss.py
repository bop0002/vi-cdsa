"""Focal Loss implementation for multi-class classification with class imbalance."""
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters
    ----------
    alpha : Optional[Union[float, Sequence[float], torch.Tensor]]
        Weighting factor for each class. If a list/tensor, must have length = num_classes.
    gamma : float, default=2.0
        Focusing parameter. Higher values give more weight to hard-to-classify examples.
    reduction : str, default="mean"
        Specifies reduction to apply to output: 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            raise TypeError(f"Unsupported alpha type: {type(alpha)}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class focal loss.
        
        inputs : torch.Tensor of shape (batch_size, num_classes) - raw unnormalized logits.
        targets : torch.Tensor of shape (batch_size,) - ground truth class indices.
        """
        # Cross entropy loss without reduction: -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # p_t: probability of true class
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1.0 - p_t) ** self.gamma
        
        loss = focal_term * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
