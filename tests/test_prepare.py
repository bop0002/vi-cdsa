import pandas as pd

from src.prepare import _merge_extra


def frame(ids, keys, label=2):
    return pd.DataFrame({"id": ids, "key": keys, "label": label})


def test_merge_extra_drops_exact_and_near_copies_keeps_base():
    base = frame(["res1_1"], ["quán ăn ngon giá cả lại hợp lý vừa với túi tiền"])
    extra = frame(["foody_1", "foody_2", "foody_3", "foody_4"], [
        "quán ăn ngon giá cả lại hợp lý vừa với túi tiền",                # exact copy
        "quán ăn ngon giá cả lại hợp lý món nem nướng tại đây rất ngon",  # same first 8 words
        "phục vụ chậm nhưng đồ ăn ổn",
        "món này ở bên A rồi",                                             # in the other domain
    ])
    out = _merge_extra(base, extra, other_keys={"món này ở bên A rồi"})
    assert list(out.id) == ["res1_1", "foody_3"]


def test_merge_extra_drops_rows_whose_id_is_in_base():
    base = frame(["tiki_1"], ["sách đẹp giao nhanh"])
    out = _merge_extra(base, frame(["tiki_1", "tiki_2"], ["tuyệt vời sách đẹp giao nhanh", "giấy mỏng chữ mờ"]), set())
    assert list(out.id) == ["tiki_1", "tiki_2"]
