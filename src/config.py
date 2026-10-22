from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
VNCORENLP_DIR = ROOT / "models" / "vncorenlp"

SEED = 42
MIN_WORDS = 3
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
# res1 gốc: 0 = negative, 1 = positive, 2 = neutral (đã kiểm chứng bằng tỷ lệ từ khen/chê)
RES1_LABEL_MAP = {0: 0, 1: 2, 2: 1}

# Max train rows per class (same for A and B); classes not listed are capped only by min(A, B).
# Neg:Neu:Pos = 2:1:2 with Neutral at its B limit (6,157): best val Macro-F1 among 15/20/25/33% Neutral (TF-IDF+LR).
TRAIN_CLASS_CAP: dict | None = {0: 12314, 2: 12314}

OUTPUT_COLUMNS = [
    "id", "domain", "source", "group_id", "rating", "label", "label_source", "split",
    "text_raw", "text_clean", "text_ml", "text_dl", "n_words",
]
