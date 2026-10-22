"""Train reproducible TF-IDF baselines using source-domain validation only."""
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import LABEL_NAMES
from src.dataset import DOMAINS, load_split

MODELS = ("logistic_regression", "linear_svc")
LABELS = list(LABEL_NAMES)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(data_dir, domain, split):
    frame = load_split(domain, split, processed_dir=data_dir)
    required = {"id", "text_ml", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{domain}_{split}: missing columns {required - set(frame.columns)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f"{domain}_{split}: empty data or missing values")
    if frame.id.duplicated().any():
        raise ValueError(f"{domain}_{split}: duplicate IDs")
    if not frame.text_ml.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
        raise ValueError(f"{domain}_{split}: invalid text_ml")
    if set(frame.label) != set(LABELS):
        raise ValueError(f"{domain}_{split}: expected all labels {LABELS}")
    return frame


def assert_disjoint(left, right, description):
    for column in ("id", "text_clean"):
        if column in left and column in right and set(left[column]) & set(right[column]):
            raise ValueError(f"Data leakage ({column}): {description}")


def train(data_dir, model_dir, config):
    """Fit vocabulary on train; choose C/weights by source validation Macro-F1.

    Keep the winning train-only fit; never refit on validation or target data.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if any(model_dir.iterdir()):
        raise FileExistsError(f"Use an empty model directory: {model_dir}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "python": platform.python_version(),
        "versions": {p: version(p) for p in ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "pyarrow")},
        "data_sha256": {}, "models": {},
        "selection": "source validation macro_f1; ties keep first candidate; no refit",
    }
    frames = {
        (d, s): read_data(data_dir, d, s)
        for d in DOMAINS for s in ("train", "val")
    }
    items = list(frames.items())
    for i, (key, frame) in enumerate(items):
        for other_key, other in items[i + 1:]:
            assert_disjoint(frame, other, f"{key} vs {other_key}")
        name = f"{key[0]}_{key[1]}.parquet"
        manifest["data_sha256"][name] = fingerprint(Path(data_dir) / name)
    for domain in DOMAINS:
        training, validation = frames[domain, "train"], frames[domain, "val"]
        params = dict(config["tfidf"])
        params["ngram_range"] = tuple(params["ngram_range"])
        start = perf_counter()
        vectorizer = TfidfVectorizer(**params)
        x_train = vectorizer.fit_transform(training.text_ml)
        x_val = vectorizer.transform(validation.text_ml)
        vectorizer_seconds = perf_counter() - start
        for model_name in MODELS:
            print(f"Training {model_name} on {domain}...", flush=True)
            best_score, best_model, best_params = -1, None, None
            trials = []
            for c in config["search"]["C"]:
                for weight in config["search"]["class_weight"]:
                    common = dict(C=c, class_weight=weight, max_iter=config["max_iter"],
                                  random_state=config["seed"])
                    estimator = (
                        LogisticRegression(solver="lbfgs", **common)
                        if model_name == "logistic_regression"
                        else LinearSVC(dual=True, **common)
                    )
                    started = perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        estimator.fit(x_train, training.label)
                    score = f1_score(validation.label, estimator.predict(x_val),
                                     labels=LABELS, average="macro", zero_division=0)
                    trial = {"C": c, "class_weight": weight,
                             "validation_macro_f1": score,
                             "fit_seconds": perf_counter() - started}
                    trials.append(trial)
                    if score > best_score:
                        best_score, best_model, best_params = score, estimator, trial
            key = f"{model_name}_{domain}"
            checkpoint = model_dir / f"{key}.joblib"
            joblib.dump(Pipeline([("tfidf", vectorizer), ("classifier", best_model)]), checkpoint)
            manifest["models"][key] = {
                "model": model_name, "train_domain": domain,
                "checkpoint_sha256": fingerprint(checkpoint),
                "selected": best_params, "trials": trials,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "vectorizer_seconds": vectorizer_seconds,
            }
    write_json(model_dir / "manifest.json", manifest)
    return manifest
