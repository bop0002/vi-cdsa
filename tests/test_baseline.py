import joblib
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.baseline import assert_disjoint, read_data, train
from src.baseline_evaluation import evaluate


@pytest.fixture
def experiment(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for domain in ("A", "B"):
        for split in ("train", "val", "test"):
            rows = []
            for label, word in enumerate(("bad", "average", "good")):
                for i in range(3):
                    unique = f"{domain.lower()}{split}unique{i}{label}"
                    rows.append({"id": unique, "text_ml": f"{word} service {unique}",
                                 "text_clean": f"{word} service {unique}", "label": label})
            frame = pd.DataFrame(rows)
            frame.to_parquet(data / f"{domain}_{split}.parquet")
            if split == "test":
                frame.groupby("label").head(2).to_parquet(data / f"{domain}_test_balanced.parquet")
    config = {"seed": 42, "tfidf": {"ngram_range": [1, 2], "max_features": 100,
                                   "min_df": 1},
              "search": {"C": [0.1, 1.0], "class_weight": [None]}, "max_iter": 1000}
    return data, tmp_path / "models", tmp_path / "results", config


def test_end_to_end_and_no_validation_vocabulary(experiment):
    data, models, results, config = experiment
    manifest = train(data, models, config)
    assert len(manifest["models"]) == 4
    for key, entry in manifest["models"].items():
        pipeline = joblib.load(models / f"{key}.joblib")
        assert not any("valunique" in token or "testunique" in token
                       for token in pipeline["tfidf"].vocabulary_)
        assert entry["selected"]["validation_macro_f1"] == max(
            trial["validation_macro_f1"] for trial in entry["trials"])
    evaluate(data, models, results)
    metrics = pd.read_csv(results / "metrics/summary.csv")
    assert len(metrics) == 16
    assert len(pd.read_csv(results / "metrics/per_class.csv")) == 48
    for _, row in metrics.iterrows():
        stem = f"{row.model}_{row.train_domain}_to_{row.test_domain}_{row.split}"
        predictions = pd.read_csv(results / "predictions" / f"{stem}.csv")
        assert row.macro_f1 == pytest.approx(f1_score(
            predictions.label, predictions.prediction, average="macro"))
        matrix = pd.read_csv(results / "confusion_matrix" / f"{stem}.csv", index_col=0)
        assert matrix.to_numpy().sum() == row.n_test
    drops = pd.read_csv(results / "metrics/performance_drop.csv")
    assert len(drops) == 8
    for row in drops.itertuples():
        assert row.performance_drop == pytest.approx(row.in_domain_macro_f1 - row.cross_domain_macro_f1)
    with pytest.raises(FileExistsError):
        train(data, models, config)
    with pytest.raises(FileExistsError):
        evaluate(data, models, results)
    frame = pd.read_parquet(data / "A_train.parquet")
    frame.loc[0, "text_ml"] = "changed"
    frame.to_parquet(data / "A_train.parquet")
    with pytest.raises(ValueError, match="data changed"):
        evaluate(data, models, results.parent / "changed")


def test_leakage_and_invalid_input(experiment):
    data, _, _, _ = experiment
    frame = read_data(data, "A", "train")
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(frame, frame, "same data")
    frame.loc[0, "text_ml"] = ""
    frame.to_parquet(data / "A_train.parquet")
    with pytest.raises(ValueError, match="text_ml"):
        read_data(data, "A", "train")
