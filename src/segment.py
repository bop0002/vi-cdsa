import os
import re
import urllib.request
from functools import lru_cache

import py_vncorenlp

from src.config import VNCORENLP_DIR

_SPACES = re.compile(r"\s+")

_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
_FILES = {
    "VnCoreNLP-1.2.jar": _BASE + "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab": _BASE + "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": _BASE + "models/wordsegmenter/wordsegmenter.rdr",
}


def _ensure_model_files() -> None:
    for rel, url in _FILES.items():
        dest = VNCORENLP_DIR / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            urllib.request.urlretrieve(url, str(tmp))
            tmp.replace(dest)


@lru_cache(maxsize=1)
def _model():
    _ensure_model_files()
    cwd = os.getcwd()
    try:
        return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(VNCORENLP_DIR))
    finally:
        os.chdir(cwd)


def segment(text: str) -> str:
    return _SPACES.sub(" ", " ".join(_model().word_segment(text))).strip()


def to_ml(seg: str) -> str:
    return _SPACES.sub(" ", seg.replace(",", " ")).strip()
