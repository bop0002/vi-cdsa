"""Compare immutable Git dataset snapshots and their baseline metrics."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.config import ROOT


def git_bytes(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def snapshot(ref):
    paths = git_bytes("ls-tree", "-r", "--name-only", ref, "data/processed").decode().splitlines()
    return {Path(p).name: git_bytes("show", f"{ref}:{p}")
            for p in paths if p.endswith(".parquet")}


def compare_frames(old, new):
    a, b = old.set_index("id"), new.set_index("id")
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError("Duplicate IDs in snapshot")
    common = a.index.intersection(b.index)
    result = {"old_rows": len(a), "new_rows": len(b), "common_ids": len(common),
              "added_ids": len(b.index.difference(a.index)),
              "removed_ids": len(a.index.difference(b.index))}
    for column in ("label", "text_raw", "text_clean", "text_ml", "source", "split"):
        if column in a and column in b:
            left, right = a.loc[common, column], b.loc[common, column]
            result[f"changed_{column}"] = int((~(left.eq(right) | (left.isna() & right.isna()))).sum())
    result["same_prediction_inputs"] = (
        len(a) == len(b) == len(common)
        and result.get("changed_label") == result.get("changed_text_ml") == 0
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ref", required=True)
    parser.add_argument("--new-ref", required=True)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    refs = {v: git_bytes("rev-parse", ref).decode().strip()
            for v, ref in (("old", args.old_ref), ("new", args.new_ref))}
    snapshots = {v: snapshot(ref) for v, ref in refs.items()}
    frames = {v: {name: pd.read_parquet(io.BytesIO(blob)) for name, blob in files.items()}
              for v, files in snapshots.items()}
    manifests = {v: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                 for v, path in (("old", args.old_results), ("new", args.new_results))}
    for v, manifest in manifests.items():
        for name, digest in {**manifest["data_sha256"], **manifest["test_sha256"]}.items():
            if hashlib.sha256(snapshots[v][name]).hexdigest() != digest:
                raise ValueError(f"{v} results do not match snapshot: {name}")
    rows, distributions, sources, schemas = [], [], [], []
    for name in sorted(set(frames["old"]) | set(frames["new"])):
        a, b = frames["old"].get(name), frames["new"].get(name)
        if a is not None and b is not None:
            row = {"file": name, **compare_frames(a, b)}
            row["same_file_bytes"] = snapshots["old"][name] == snapshots["new"][name]
        else:
            row = {"file": name, "old_rows": 0 if a is None else len(a),
                   "new_rows": 0 if b is None else len(b), "status": "added" if a is None else "removed"}
        rows.append(row)
        for v, frame in (("old", a), ("new", b)):
            if frame is None:
                continue
            schemas.append({"version": v, "file": name,
                            "schema": {c: str(t) for c, t in frame.dtypes.items()}})
            for label, count in frame.label.value_counts().sort_index().items():
                distributions.append({"version": v, "file": name, "label": label,
                                      "count": count, "proportion": count / len(frame)})
            for source, count in frame.source.value_counts().items():
                sources.append({"version": v, "file": name, "source": source, "count": count})
    # Main split union identifies movement between splits separately from additions.
    movements = []
    for domain in ("A", "B"):
        combined = {v: pd.concat([frames[v][f"{domain}_{s}.parquet"].assign(split=s)
                                 for s in ("train", "val", "test")]) for v in frames}
        movements.append({"domain": domain, **compare_frames(combined["old"], combined["new"])})
    pd.DataFrame(rows).to_csv(out / "dataset_changes.csv", index=False)
    pd.DataFrame(distributions).to_csv(out / "label_distribution.csv", index=False)
    pd.DataFrame(sources).to_csv(out / "source_distribution.csv", index=False)
    pd.DataFrame(movements).to_csv(out / "domain_changes.csv", index=False)
    tables = {v: pd.read_csv(path / "metrics/summary.csv") for v, path in
              (("old", args.old_results), ("new", args.new_results))}
    keys = ["model", "train_domain", "test_domain", "split", "setting"]
    delta = tables["old"].merge(tables["new"], on=keys, suffixes=("_old", "_new"),
                                validate="one_to_one", how="outer", indicator=True)
    if not delta["_merge"].eq("both").all():
        raise ValueError("Experiment matrices do not match")
    delta = delta.drop(columns="_merge")
    for metric in ("macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1"):
        delta[f"delta_{metric}"] = delta[f"{metric}_new"] - delta[f"{metric}_old"]
    delta.to_csv(out / "metrics_changes.csv", index=False)
    provenance = {"git_refs": refs, "same_config": manifests["old"]["config"] == manifests["new"]["config"],
                  "same_library_versions": manifests["old"]["versions"] == manifests["new"]["versions"],
                  "training": "Fresh TF-IDF and estimators; no loading of old checkpoints during training",
                  "schemas": schemas}
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# So sánh dữ liệu và mô hình", "", f"Dữ liệu cũ: `{refs['old']}`.",
             f"Dữ liệu mới: `{refs['new']}`.", "",
             "Train mới từ đầu: TF-IDF và classifier mới hoàn toàn; không warm-start, không partial_fit.",
             f"Cùng cấu hình: {provenance['same_config']}; cùng phiên bản thư viện: {provenance['same_library_versions']}.",
             "Dùng train chuẩn cho cả hai phiên bản; train_balanced mới chỉ được thống kê, không dùng trong lần train này.", "",
             "| Tệp | Số dòng cũ | Số dòng mới | ID thêm | ID bỏ | Nhãn đổi (ID chung) | Text ML đổi (ID chung) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['old_rows']} | {r['new_rows']} | {r.get('added_ids', '—')} | {r.get('removed_ids', '—')} | {r.get('changed_label', '—')} | {r.get('changed_text_ml', '—')} |")
    stable = all(r.get("same_prediction_inputs", False) for r in rows if "_test" in r["file"])
    lines += ["", f"Các tập test giữ nguyên ID, text_ml và nhãn: {stable}.",
              "Đối chiếu theo ID, không phụ thuộc thứ tự dòng. SHA-256 kết quả đã được đối chiếu với snapshot Git.",
              "File Parquet thay đổi byte không nhất thiết có nghĩa nội dung đầu vào mô hình thay đổi.", ""]
    lines += ["## Phân bố train", "", "| Phiên bản | Miền | Negative | Neutral | Positive |",
              "|---|---|---:|---:|---:|"]
    for v in ("old", "new"):
        for domain in ("A", "B"):
            counts = frames[v][f"{domain}_train.parquet"].label.value_counts()
            lines.append(f"| {v} | {domain} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} |")
    lines += ["", "## Nguồn train", "", "| Phiên bản | Tệp | Nguồn | Số mẫu |", "|---|---|---|---:|"]
    for r in sources:
        if r["file"].endswith("_train.parquet"):
            lines.append(f"| {r['version']} | {r['file']} | {r['source']} | {r['count']} |")
    lines.append("")
    for split in ("test", "test_balanced"):
        lines += [f"## {split}", "", "| Mô hình | Chiều | Macro-F1 cũ | Macro-F1 mới | Chênh lệch mới − cũ |",
                  "|---|---|---:|---:|---:|"]
        for r in delta[delta.split == split].itertuples():
            lines.append(f"| {r.model} | {r.train_domain}→{r.test_domain} | {r.macro_f1_old:.4f} | {r.macro_f1_new:.4f} | {r.delta_macro_f1:+.4f} |")
        lines.append("")
    lines += ["## Giới hạn diễn giải", "",
              "Đây là so sánh điểm số một seed, chưa phải kiểm định ý nghĩa thống kê.",
              "Thay đổi đồng thời quy mô, nguồn và tỷ lệ nhãn train nên không quy chênh lệch cho riêng một yếu tố.",
              "Tham số được chọn lại trên validation theo cùng lưới tìm kiếm.",
              "Test cân bằng là tập con test gốc; không coi hai bảng là hai kiểm chứng độc lập."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
