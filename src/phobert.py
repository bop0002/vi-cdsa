"""PhoBERT pipeline for Sentiment Analysis (Training, Evaluation, In-Domain & Cross-Domain)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.focal_loss import FocalLoss

DEFAULT_MODEL_NAME = "vinai/phobert-base-v2"
NUM_CLASSES = 3
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


class ReviewDataset(Dataset):
    """PyTorch Dataset for text reviews."""

    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[int]] = None):
        self.texts = list(texts)
        self.labels = [int(y) for y in labels] if labels is not None else None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": str(self.texts[idx])}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class DynamicPaddingCollator:
    """Collator that tokenizes and dynamically pads batch to the longest sequence."""

    def __init__(self, tokenizer: Any, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            encoded["labels"] = labels
        return encoded


class SentimentPhoBERT(nn.Module):
    """PhoBERT Backbone with classification head."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Using [CLS] token representation (token at index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_token))
        return logits


def compute_class_weights(labels: Union[Sequence[int], np.ndarray, pd.Series]) -> torch.Tensor:
    """Compute balanced class weights: w_c = N / (C * N_c)."""
    labels_arr = np.array(labels, dtype=int)
    classes, counts = np.unique(labels_arr, return_counts=True)
    total_samples = len(labels_arr)
    num_classes = len(classes)
    weights = total_samples / (num_classes * counts.astype(float))
    # Normalized so mean weight is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_loss_function(
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> nn.Module:
    """Return configured loss function: 'cross_entropy' or 'focal'."""
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, reduction="mean")
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'cross_entropy' or 'focal'.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    return_details: bool = False,
) -> Dict[str, Any]:
    """Evaluate model performance on a dataloader.
    
    Computes Loss, Macro-F1, Accuracy, Precision, Recall, Per-class F1, and Confusion Matrix.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch.get("labels")

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_preds.extend(preds)
        all_probs.extend(probs)

        if targets is not None:
            targets = targets.to(device)
            if criterion is not None:
                loss = criterion(logits, targets)
                total_loss += loss.item()
            all_targets.extend(targets.cpu().numpy())

    has_targets = len(all_targets) > 0
    results: Dict[str, Any] = {}

    if has_targets:
        targets_arr = np.array(all_targets)
        preds_arr = np.array(all_preds)

        results["loss"] = total_loss / max(len(dataloader), 1) if criterion else 0.0
        results["accuracy"] = float(accuracy_score(targets_arr, preds_arr))
        results["macro_f1"] = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
        results["macro_precision"] = float(precision_score(targets_arr, preds_arr, average="macro", zero_division=0))
        results["macro_recall"] = float(recall_score(targets_arr, preds_arr, average="macro", zero_division=0))

        # Per-class F1
        per_class_f1 = f1_score(targets_arr, preds_arr, average=None, labels=[0, 1, 2], zero_division=0)
        results["f1_neg"] = float(per_class_f1[0])
        results["f1_neu"] = float(per_class_f1[1])
        results["f1_pos"] = float(per_class_f1[2])

        # Confusion Matrix
        cm = confusion_matrix(targets_arr, preds_arr, labels=[0, 1, 2])
        results["confusion_matrix"] = cm.tolist()

    if return_details:
        results["preds"] = np.array(all_preds)
        results["probs"] = np.array(all_probs)
        if has_targets:
            results["targets"] = np.array(all_targets)

    return results


def train_phobert(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, float, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Train PhoBERT model with evaluation after each epoch, saving best checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = get_loss_function(loss_type=loss_type, class_weights=class_weights, gamma=gamma)
    criterion.to(device)

    # Optimizer with weight decay exclusion for bias & LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_macro_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    history = []

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_metrics = evaluate_model(model=model, dataloader=val_loader, criterion=criterion, device=device)
        val_macro_f1 = val_metrics["macro_f1"]

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_neu": val_metrics["f1_neu"],
        }
        history.append(log_entry)

        if progress_callback:
            progress_callback(epoch, train_loss, val_metrics)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_metrics = val_metrics
            if save_path:
                torch.save(model.state_dict(), save_path)

    return {
        "best_macro_f1": best_macro_f1,
        "best_metrics": best_metrics,
        "history": history,
    }
