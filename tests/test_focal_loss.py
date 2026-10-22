import pytest
import torch

from src.focal_loss import FocalLoss


def test_focal_loss_shape_and_reduction():
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    logits = torch.randn(8, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0

    # Backward pass check
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_focal_loss_with_alpha():
    alpha = [1.0, 2.0, 0.5]
    loss_fn = FocalLoss(alpha=alpha, gamma=1.5, reduction="none")
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])

    loss = loss_fn(logits, targets)
    assert loss.shape == (4,)
    assert (loss >= 0).all()


def test_focal_loss_gamma_zero_matches_ce():
    # When gamma=0 and alpha=None, FocalLoss should match standard CrossEntropy
    focal = FocalLoss(gamma=0.0, reduction="mean")
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))

    ce_val = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    fl_val = focal(logits, targets)

    assert torch.allclose(ce_val, fl_val, atol=1e-5)
