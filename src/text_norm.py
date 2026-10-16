import re
import unicodedata

import emoji

AUTO_TITLE = re.compile(
    r"^\s*(rất không hài lòng|không hài lòng|cực k[ìỳ] hài lòng|hài lòng|bình thường)"
    r"\s*(?:[.!:\-]+\s*|$)",
    re.IGNORECASE,
)
NO_SCORE = re.compile(r"\(\*\)\s*đánh giá không tính điểm", re.IGNORECASE)
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SENT_PUNCT = re.compile(r"[.!?;:…\n\r\t]+")
OTHER_CHARS = re.compile(r"[^\w\s,/]")
REPEAT_LETTER = re.compile(r"([^\W\d_])\1{2,}")
COMMAS = re.compile(r"\s*,[\s,]*")
SPACES = re.compile(r"\s+")

TEENCODE = {
    "k": "không", "ko": "không", "kh": "không", "hk": "không", "hok": "không",
    "khg": "không", "k0": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "sp": "sản phẩm", "sd": "sử dụng", "nv": "nhân viên", "đt": "điện thoại",
    "j": "gì", "vs": "với", "mn": "mọi người", "mng": "mọi người",
    "mik": "mình", "mk": "mình", "mh": "mình", "r": "rồi", "lun": "luôn",
    "wa": "quá", "ng": "người", "cx": "cũng", "cg": "cũng", "trc": "trước",
    "nhìu": "nhiều", "bik": "biết", "bít": "biết", "đag": "đang",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
}
WORD = re.compile(r"\b\w+\b")
# "kg" = "không" (phủ định), TRỪ khi đứng sau số ("2 kg", "6kg" là cân nặng).
# "6kg" là một token nên WORD không bao giờ khớp riêng "kg"; chỉ cần chặn trường hợp "2 kg".
KG_AS_NEGATION = re.compile(r"(?<!\d)(?<!\d )\bkg\b")


def strip_platform_text(t: str) -> str:
    t = NO_SCORE.sub(" ", t)
    return AUTO_TITLE.sub("", t, count=1)


def normalize_format(t: str) -> str:
    t = unicodedata.normalize("NFC", t).lower()
    t = URL.sub(" ", t)
    t = emoji.replace_emoji(t, " ")
    t = SENT_PUNCT.sub(", ", t)
    t = t.replace("_", " ")
    t = OTHER_CHARS.sub(" ", t)
    t = REPEAT_LETTER.sub(r"\1\1", t)
    t = COMMAS.sub(", ", t)
    t = SPACES.sub(" ", t)
    return t.strip(" ,")


def apply_teencode(t: str) -> str:
    t = KG_AS_NEGATION.sub("không", t)
    return WORD.sub(lambda m: TEENCODE.get(m.group(0), m.group(0)), t)


def clean_text(t: str) -> str:
    return apply_teencode(normalize_format(strip_platform_text(t)))
