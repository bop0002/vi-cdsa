import pandas as pd
import pytest

from src.error_analysis import (
    classify_error_cause,
    extract_error_cases,
    format_error_table_markdown,
)


def test_classify_error_cause_short_review():
    etype, reason = classify_error_cause(
        text="quá tệ",
        true_label=0,
        pred_label=2,
        source_domain="A",
        target_domain="B",
    )
    assert etype == "Review quá ngắn"


def test_classify_error_cause_domain_specific():
    etype, reason = classify_error_cause(
        text="món ăn ở đây nêm nếm rất nhạt, nhân viên phục vụ chậm chạp",
        true_label=0,
        pred_label=2,
        source_domain="A",
        target_domain="B",
    )
    assert etype == "Domain-specific vocabulary"


def test_classify_error_cause_mixed_sentiment():
    etype, reason = classify_error_cause(
        text="đồ ăn ngon nhưng phục vụ bàn rất khó chịu",
        true_label=0,
        pred_label=2,
        source_domain="A",
        target_domain="B",
    )
    assert etype == "Mixed Sentiment"


def test_extract_error_cases_and_markdown():
    data = [
        {"id": "1", "text_clean": "quá dở", "label": 0, "pred": 2, "rating": 1.0},
        {"id": "2", "text_clean": "sách bìa đẹp nhưng gáy bị bung keo", "label": 0, "pred": 2, "rating": 2.0},
        {"id": "3", "text_clean": "món ăn tuyệt vời nhưng không gian hơi chật", "label": 1, "pred": 2, "rating": 3.0},
    ]
    df = pd.DataFrame(data)
    errors = extract_error_cases(df, source_domain="A", target_domain="B", max_cases_per_type=2)

    assert len(errors) == 3
    assert "error_type" in errors.columns

    md = format_error_table_markdown(errors)
    assert "| STT |" in md
    assert "quá dở" in md
