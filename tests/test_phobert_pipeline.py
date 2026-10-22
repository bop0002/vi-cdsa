import numpy as np
import pytest
import torch

from src.phobert import (
    DynamicPaddingCollator,
    ReviewDataset,
    compute_class_weights,
    get_loss_function,
)


def test_review_dataset():
    texts = ["sách hay", "quán dở", "bình thường"]
    labels = [2, 0, 1]
    dataset = ReviewDataset(texts=texts, labels=labels)

    assert len(dataset) == 3
    item = dataset[0]
    assert item["text"] == "sách hay"
    assert item["label"] == 2


def test_compute_class_weights():
    # 10 neg, 2 neu, 20 pos (imbalanced)
    labels = [0] * 10 + [1] * 2 + [2] * 20
    weights = compute_class_weights(labels)

    assert len(weights) == 3
    # Minority class (1) should receive the highest weight
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]


def test_get_loss_function():
    ce = get_loss_function("cross_entropy")
    assert isinstance(ce, torch.nn.CrossEntropyLoss)

    fl = get_loss_function("focal", gamma=2.0)
    assert hasattr(fl, "gamma")
    assert fl.gamma == 2.0


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        batch_size = len(texts)
        seq_len = min(max_length, 8)
        return {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }


def test_dynamic_padding_collator():
    tok = DummyTokenizer()
    collator = DynamicPaddingCollator(tokenizer=tok, max_length=32)

    batch = [
        {"text": "cuốn sách này rất bổ ích", "label": 2},
        {"text": "đồ ăn tệ", "label": 0},
    ]

    out = collator(batch)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert "labels" in out
    assert out["input_ids"].shape[0] == 2
    assert out["labels"].tolist() == [2, 0]
