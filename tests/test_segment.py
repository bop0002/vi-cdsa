from src.segment import segment, to_ml


def test_to_ml_drops_commas():
    assert to_ml("giao_hàng nhanh , sách đẹp") == "giao_hàng nhanh sách đẹp"


def test_segment_joins_compound_words():
    out = segment("giao hàng nhanh, nhân viên nhiệt tình, đóng gói cẩn thận")
    assert "nhân_viên" in out and "nhiệt_tình" in out and "đóng_gói" in out


def test_segment_single_line_output():
    assert "\n" not in segment("sách hay, giao nhanh, đóng gói kỹ")
