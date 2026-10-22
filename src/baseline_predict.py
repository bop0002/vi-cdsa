"""Inference on raw comments with the existing training preprocessing."""
import json
import os
import shutil
import tempfile
from pathlib import Path
from functools import lru_cache

import joblib

from src.baseline import MODELS, fingerprint
from src.config import ROOT, VNCORENLP_DIR
from src.dataset import DOMAINS

LABELS_VI = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
MAX_COMMENT_LENGTH = 5000


@lru_cache(maxsize=1)
def demo_segmenter():
    # VnCoreNLP does not decode accented/space-containing model URLs correctly.
    # Keep the original resources intact; use an identical ASCII-path copy.
    import py_vncorenlp
    from src.segment import _FILES
    directory = VNCORENLP_DIR
    if not str(directory).isascii() or " " in str(directory):
        directory = Path(tempfile.mkdtemp(prefix="vi_cdsa_vncorenlp_"))
        if not str(directory).isascii() or " " in str(directory):
            raise RuntimeError("VnCoreNLP cần đường dẫn TEMP không dấu và không có khoảng trắng.")
        for name in _FILES:
            dest = directory / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(VNCORENLP_DIR / name, dest)
        import atexit
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
    cwd = os.getcwd()
    try:
        return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(directory))
    finally:
        os.chdir(cwd)


@lru_cache(maxsize=4)
def load_model(model, domain):
    if model not in MODELS or domain not in DOMAINS:
        raise ValueError("Mô hình hoặc miền train không hợp lệ.")
    directory = ROOT / "models" / "baseline"
    key = f"{model}_{domain}"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / f"{key}.joblib"
    if fingerprint(path) != manifest["models"][key]["checkpoint_sha256"]:
        raise ValueError("Checkpoint không khớp manifest. Hãy kiểm tra lại file mô hình.")
    return joblib.load(path)


def prepare_comment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Vui lòng nhập bình luận.")
    if len(text) > MAX_COMMENT_LENGTH:
        raise ValueError(f"Bình luận tối đa {MAX_COMMENT_LENGTH} ký tự.")
    from src.text_norm import clean_text
    from src.segment import to_ml, _FILES

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Bình luận không còn nội dung sau khi chuẩn hóa. Hãy nhập câu cụ thể hơn.")
    if any(not (VNCORENLP_DIR / name).is_file() for name in _FILES):
        raise FileNotFoundError(
            "Thiếu tài nguyên tách từ. Chạy: python -m src.baseline_demo --prepare"
        )
    result = to_ml(" ".join(demo_segmenter().word_segment(cleaned)))
    if not result:
        raise ValueError("Không tìm thấy nội dung để đánh giá.")
    return result


def predict_comment(text, model, domain):
    pipeline = load_model(model, domain)
    prepared = prepare_comment(text)
    features = pipeline["tfidf"].transform([prepared])
    if features.nnz == 0:
        raise ValueError("Bình luận không có từ nằm trong từ vựng mô hình. Hãy nhập câu rõ hơn.")
    label = int(pipeline["classifier"].predict(features)[0])
    return {"label": label, "label_name": LABELS_VI[label], "text_ml": prepared,
            "model": model, "domain": domain}
