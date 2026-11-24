import numpy as np
import pandas as pd
import pytest
from src.split import balanced_subset, frozen_group_split, full_split, group_split


def synthetic(n_groups=200, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for j in range(per_group):
            rows.append({"id": f"{g}_{j}", "group_id": f"g{g}",
                         "label": int(rng.choice([0, 1, 2], p=[0.1, 0.1, 0.8]))})
    return pd.DataFrame(rows)


def test_group_split_no_group_leak_and_ratios():
    df = group_split(synthetic())
    groups = {s: set(df[df.split == s].group_id) for s in ["train", "val", "test"]}
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"]) \
        and not (groups["val"] & groups["test"])
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.03 and abs(frac["test"] - 0.1) < 0.03


def test_group_split_deterministic():
    assert group_split(synthetic()).split.tolist() == group_split(synthetic()).split.tolist()


def test_balanced_subset_equal_classes():
    t = pd.DataFrame({"id": range(100), "label": [0] * 10 + [1] * 20 + [2] * 70})
    out = balanced_subset(t)
    assert out.label.value_counts().tolist() == [10, 10, 10]


def test_frozen_group_split_uses_listed_groups(tmp_path):
    p = tmp_path / "held.csv"
    pd.DataFrame({"group_id": ["g1", "g2"], "split": ["val", "test"]}).to_csv(p, index=False)
    df = pd.DataFrame({"group_id": ["g1", "g2", "g3", "g1"], "label": [0, 1, 2, 2]})
    assert frozen_group_split(df, p)["split"].tolist() == ["val", "test", "train", "val"]


def test_frozen_group_split_falls_back_without_file(tmp_path):
    assert frozen_group_split(synthetic(), tmp_path / "nope.csv").split.tolist() == group_split(synthetic()).split.tolist()


def test_full_split_keeps_each_domain_prior_same_size_no_group_leak():
    a = synthetic(n_groups=300, seed=1).assign(id=lambda d: "a" + d.id, group_id=lambda d: "a" + d.group_id)
    b = synthetic(n_groups=250, seed=2).assign(id=lambda d: "b" + d.id, group_id=lambda d: "b" + d.group_id)
    b = b.assign(label=lambda d: d.label.where(d.index % 4 != 0, 0))  # B gets a different class prior
    oa, ob = full_split(a, b)
    assert oa.split.value_counts().to_dict() == ob.split.value_counts().to_dict()  # same size in every split
    for src, out in zip((a, b), (oa, ob)):
        frac = out.split.value_counts(normalize=True)
        assert abs(frac["train"] - 0.8) < 0.01 and abs(frac["val"] - 0.1) < 0.01 and abs(frac["test"] - 0.1) < 0.01
        prior = src.label.value_counts(normalize=True)
        for s in ("train", "val", "test"):
            got = out[out.split == s].label.value_counts(normalize=True)
            assert (got - prior).abs().max() < 0.02
        assert out.groupby("group_id").split.nunique().max() == 1
        assert not out.id.duplicated().any()
