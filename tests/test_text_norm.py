import pytest
from src.text_norm import strip_platform_text, normalize_format, apply_teencode, clean_text


@pytest.mark.parametrize("raw, expected", [
    ("Cực kì hài lòng. Sách đẹp", "Sách đẹp"),
    ("CỰC KỲ HÀI LÒNG! Sách đẹp", "Sách đẹp"),
    ("Không hài lòng!!! bìa rách", "bìa rách"),
    ("Rất không hài lòng", ""),
    ("Hài lòng với cách đóng gói", "Hài lòng với cách đóng gói"),
    ("Sách ổn (*) Đánh giá không tính điểm", "Sách ổn"),
])
def test_strip_platform_text(raw, expected):
    assert strip_platform_text(raw).strip() == expected


@pytest.mark.parametrize("raw, expected", [
    ("Sách HAY quá!!! 😍\nGiao nhanh.", "sách hay quá, giao nhanh"),
    ("Đọc xong thấy: \"tuyệt\" (thật sự)", "đọc xong thấy, tuyệt thật sự"),
    ("ngonnnnn quá giá 1000đ", "ngonn quá giá 1000đ"),
    ("xem https://tiki.vn/abc nhé", "xem nhé"),
    ("khách_sạn đẹp ,  , sạch", "khách sạn đẹp, sạch"),
    ("40k/suất , no", "40k/suất, no"),
    ("hoà bình", "hoà bình"),  # NFD -> NFC
])
def test_normalize_format(raw, expected):
    assert normalize_format(raw) == expected


def test_apply_teencode_only_whole_words():
    assert apply_teencode("sp ok, k thích, giá 40k, đc giao nhanh") == \
        "sản phẩm ok, không thích, giá 40k, được giao nhanh"


@pytest.mark.parametrize("raw, expected", [
    ("lang mang kg cụ thể", "lang mang không cụ thể"),       # kg = không (36/40 dòng ở book)
    ("kg hiểu sao bìa bị trầy", "không hiểu sao bìa bị trầy"),
    ("mới tăng có 2 kg thôi", "mới tăng có 2 kg thôi"),      # cân nặng: giữ nguyên
    ("đã giảm được 6kg rồi", "đã giảm được 6kg rồi"),
    ("tầm 200k 1 kg, cốm thơm", "tầm 200k 1 kg, cốm thơm"),
])
def test_kg_context(raw, expected):
    assert apply_teencode(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("coca 27 k, sữa chua", "coca 27 k, sữa chua"),                       # giá: giữ "k"
    ("xôi thịt là 13 k, còn", "xôi thịt là 13 k, còn"),
    ("lạc và rau mà 35 k", "lạc và rau mà 35 k"),                         # cuối câu
    ("quyển 1 k hiểu sao bị bẩn", "quyển 1 không hiểu sao bị bẩn"),       # phủ định
    ("tập 2 k bị gì", "tập 2 không bị gì"),
    ("k thích lắm", "không thích lắm"),
    ("giá 40k, ok", "giá 40k, ok"),
])
def test_k_after_number_context(raw, expected):
    assert apply_teencode(raw) == expected


def test_clean_text_full_pipeline():
    assert clean_text("Cực kì hài lòng. SP đẹp lắm ko chê đc 👍") == "sản phẩm đẹp lắm không chê được"


def test_normalize_is_idempotent_on_res1_style():
    s = "mình rất thích nước dùng ở đây, món trộn ăn không hợp lắm, quán sạch sẽ"
    assert normalize_format(s) == s


def test_strip_platform_text_handles_nfd_input():
    import unicodedata
    raw = unicodedata.normalize("NFD", "Hài lòng. Sách đẹp")
    out = strip_platform_text(raw).strip()
    assert unicodedata.normalize("NFC", out) == "Sách đẹp"


def test_normalize_format_folds_math_alphanumerics():
    assert normalize_format("𝐇𝐚𝐲 lắm") == "hay lắm"
