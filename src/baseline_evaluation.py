"""Evaluate saved baselines and write comparable, traceable experiment results."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.baseline import LABELS, assert_disjoint, fingerprint, read_data, write_json
from src.config import LABEL_NAMES
from src.dataset import DOMAINS


def evaluate(data_dir, model_dir, output_dir):
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    # Refuse changed training inputs: metrics must describe the recorded experiment.
    for name, digest in manifest["data_sha256"].items():
        if fingerprint(Path(data_dir) / name) != digest:
            raise ValueError(f"Training/validation data changed: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty results directory: {output_dir}")
    for folder in ("metrics", "confusion_matrix", "predictions", "figures"):
        (output_dir / folder).mkdir()
    source_frames = [read_data(data_dir, d, s) for d in DOMAINS for s in ("train", "val")]
    tests = {(d, s): read_data(data_dir, d, s)
             for d in DOMAINS for s in ("test", "test_balanced")}
    assert_disjoint(tests["A", "test"], tests["B", "test"], "A_test vs B_test")
    for (domain, split), frame in tests.items():
        for source in source_frames:
            assert_disjoint(source, frame, f"train/val vs {domain}_{split}")
        if split == "test_balanced":
            original = tests[domain, "test"].set_index("id")
            subset = frame.set_index("id")
            if not subset.index.isin(original.index).all():
                raise ValueError(f"{domain}: balanced test must be a test subset")
            if not subset[["text_ml", "label"]].equals(original.loc[subset.index, ["text_ml", "label"]]):
                raise ValueError(f"{domain}: balanced test differs from original")
            if frame.label.value_counts().nunique() != 1:
                raise ValueError(f"{domain}: test_balanced is not balanced")
    rows, class_rows = [], []
    for key, entry in manifest["models"].items():
        checkpoint = model_dir / f"{key}.joblib"
        if fingerprint(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {key}")
        pipeline = joblib.load(checkpoint)
        for (target, split), frame in tests.items():
            started = perf_counter()
            predicted = pipeline.predict(frame.text_ml)
            seconds = perf_counter() - started
            report = classification_report(frame.label, predicted, labels=LABELS,
                                           target_names=list(LABEL_NAMES.values()),
                                           output_dict=True, zero_division=0)
            identity = {"model": entry["model"], "train_domain": entry["train_domain"],
                        "test_domain": target, "split": split,
                        "setting": "in_domain" if target == entry["train_domain"] else "cross_domain"}
            rows.append({**identity, "n_test": len(frame), "accuracy": report["accuracy"],
                         "macro_f1": report["macro avg"]["f1-score"],
                         "macro_precision": report["macro avg"]["precision"],
                         "macro_recall": report["macro avg"]["recall"],
                         "weighted_f1": report["weighted avg"]["f1-score"],
                         "predict_seconds": seconds})
            for label, name in LABEL_NAMES.items():
                class_rows.append({**identity, "label": label, "label_name": name, **report[name]})
            stem = f"{key}_to_{target}_{split}"
            write_json(output_dir / "metrics" / f"{stem}.json", report)
            columns = [c for c in ("id", "text_ml", "label") if c in frame]
            predictions = frame[columns].copy()
            predictions["prediction"] = predicted
            predictions["correct"] = predictions.label == predicted
            predictions.to_csv(output_dir / "predictions" / f"{stem}.csv", index=False)
            matrix = confusion_matrix(frame.label, predicted, labels=LABELS)
            pd.DataFrame(matrix, index=LABEL_NAMES.values(), columns=LABEL_NAMES.values()).to_csv(
                output_dir / "confusion_matrix" / f"{stem}.csv", index_label="true_label")
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(matrix, display_labels=list(LABEL_NAMES.values())).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{entry['model']}: {entry['train_domain']} → {target}\n{split}")
            fig.tight_layout()
            fig.savefig(output_dir / "confusion_matrix" / f"{stem}.png", dpi=160)
            plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "metrics" / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "metrics" / "per_class.csv", index=False)
    manifest["test_sha256"] = {f"{d}_{s}.parquet": fingerprint(Path(data_dir) / f"{d}_{s}.parquet")
                               for d, s in tests}
    write_json(output_dir / "manifest.json", manifest)
    summarize(output_dir)


def write_readable_summary(output_dir):
    """Render existing metrics for reading without changing the source CSV."""
    output_dir = Path(output_dir)
    metrics = pd.read_csv(output_dir / "metrics" / "summary.csv")
    names = {"logistic_regression": "Logistic Regression", "linear_svc": "LinearSVC"}
    directions = [("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")]
    lines = ["# Kết quả đánh giá mô hình", "",
             f"Bộ kết quả: `{output_dir.name}` · Nguồn: [summary.csv](summary.csv).", "",
             "A = đánh giá sách Tiki; B = đánh giá nhà hàng. Chiều đọc: **miền train → miền test**.",
             "A→A, B→B: cùng miền. A→B, B→A: khác miền.", "",
             "Điểm số hiển thị trên thang **0–100%**, làm tròn 2 chữ số thập phân.",
             "**Accuracy** là tỷ lệ dự đoán đúng. **Macro-F1** là trung bình F1 của ba lớp, không phải tỷ lệ dự đoán đúng.",
             "CSV gốc giữ nguyên độ chính xác số liệu. Không train hoặc đánh giá lại để tạo bảng này.", ""]
    for split, title in (("test", "Test gốc"), ("test_balanced", "Test cân bằng")):
        subset = metrics[metrics.split == split]
        lines += [f"## {title}", "",
                  "### So sánh nhanh: Accuracy và Macro-F1", "",
                  "| Train → Test | Số mẫu test | LR · Accuracy | SVC · Accuracy | LR · Macro-F1 | SVC · Macro-F1 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for source, target in directions:
            pair = subset[(subset.train_domain == source) & (subset.test_domain == target)].set_index("model")
            lr, svc = pair.loc["logistic_regression"], pair.loc["linear_svc"]
            cells = [f"{row[metric] * 100:.2f}%" for metric in ("accuracy", "macro_f1") for row in (lr, svc)]
            lines.append(f"| {source}→{target} | {int(lr.n_test):,} | " + " | ".join(cells) + " |")
        for model, name in names.items():
            lines += ["", f"### {name} · Chi tiết", "",
                      "| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |",
                      "|---|---:|---:|---:|---:|"]
            for source, target in directions:
                row = subset[(subset.model == model) & (subset.train_domain == source)
                             & (subset.test_domain == target)].iloc[0]
                lines.append(f"| {source}→{target} | {row.macro_precision * 100:.2f}% | "
                             f"{row.macro_recall * 100:.2f}% | {row.weighted_f1 * 100:.2f}% | "
                             f"{row.predict_seconds:.3f} giây |")
        lines.append("")
    lines += ["## Cách đọc các chỉ số", "",
              "- **Precision (macro):** precision trung bình của ba lớp, mỗi lớp có trọng số bằng nhau.",
              "- **Recall (macro):** recall trung bình của ba lớp, mỗi lớp có trọng số bằng nhau.",
              "- **F1 (weighted):** F1 trung bình có trọng số theo số mẫu thật của từng lớp.",
              "- **Test cân bằng:** tập con của test gốc với số mẫu bằng nhau giữa ba lớp; đọc riêng hai bảng.",
              "- Điểm từng lớp nằm trong [per_class.csv](per_class.csv). Chênh lệch điểm chưa phải kết luận về ý nghĩa thống kê.", ""]
    (output_dir / "metrics" / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(output_dir):
    output_dir = Path(output_dir)
    metrics = pd.read_csv(output_dir / "metrics" / "summary.csv")
    write_readable_summary(output_dir)
    drops = []
    for (model, source, split), group in metrics.groupby(["model", "train_domain", "split"]):
        within = group.loc[group.test_domain == source, "macro_f1"].item()
        cross = group.loc[group.test_domain != source, "macro_f1"].item()
        drops.append({"model": model, "train_domain": source, "split": split,
                      "in_domain_macro_f1": within, "cross_domain_macro_f1": cross,
                      "performance_drop": within - cross})
    pd.DataFrame(drops).to_csv(output_dir / "metrics" / "performance_drop.csv", index=False)
    text = ["# Kết quả Machine Learning Baseline", "",
            "Macro-F1 là chỉ số chính. Các điểm số ở thang 0–1.",
            "Tham số được chọn trên validation của miền train; TF-IDF chỉ fit trên train.",
            "Performance Drop = F1 cùng miền − F1 khác miền, giữ nguyên miền train.",
            "Test cân bằng là tập con của test gốc, không phải thí nghiệm độc lập.",
            "Các kết quả là ước lượng điểm của một split/seed; chưa kiểm định ý nghĩa thống kê.", ""]
    for split in ("test", "test_balanced"):
        subset = metrics[metrics.split == split].copy()
        subset["direction"] = subset.train_domain + "→" + subset.test_domain
        table = subset.pivot(index="model", columns="direction", values="macro_f1").reindex(
            columns=["A→A", "B→B", "A→B", "B→A"])
        table.to_csv(output_dir / "metrics" / f"comparison_{split}.csv")
        text += [f"## {split}", "", "| Model | A→A | B→B | A→B | B→A |",
                 "|---|---:|---:|---:|---:|"]
        for model, row in table.iterrows():
            text.append(f"| {model} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        text.append("")
        ax = table.T.plot.bar(figsize=(9, 5), ylim=(0, 1), rot=0, ylabel="Macro-F1", title=split)
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / "figures" / f"comparison_{split}.png", dpi=160)
        plt.close(ax.figure)
    text += ["## Performance Drop", "",
             "| Model | Miền train | Tập test | Drop Macro-F1 |",
             "|---|---|---|---:|"]
    for row in drops:
        text.append(f"| {row['model']} | {row['train_domain']} | {row['split']} | "
                    f"{row['performance_drop']:.4f} |")
    text.append("")
    (output_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
