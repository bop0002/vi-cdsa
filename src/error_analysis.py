"""Error Analysis module for Cross-Domain Sentiment Analysis (Section 16 of Project Plan)."""
import re
from typing import Any, Dict, List, Optional

import pandas as pd

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

DOMAIN_A_KEYWORDS = {
    "sách", "bìa", "giấy", "gáy", "trang", "tác giả", "cốt truyện", "nội dung",
    "đóng gói", "giao hàng", "ship", "tiki", "nhà xuất bản", "nxb", "bookmark",
    "màng co", "truyện", "tiểu thuyết", "đọc", "in", "chữ", "cuốn", "tập"
}

DOMAIN_B_KEYWORDS = {
    "quán", "món", "ăn", "uống", "phục vụ", "nhân viên", "bàn", "menu", "gọi món",
    "không gian", "chỗ ngồi", "vị trí", "gửi xe", "nêm nếm", "vị", "nước", "trà",
    "giá cả", "tính tiền", "hóa đơn", "bill", "order", "view", "máy lạnh"
}

NEGATION_KEYWORDS = {
    "không", "chẳng", "chả", "chưa", "kém", "mất", "dở", "tệ", "thất vọng",
    "không hề", "không được", "chẳng có", "chả có", "chưa tốt", "không ngon"
}

CONTRAST_KEYWORDS = {"nhưng", "tuy nhiên", "mặc dù", "có điều", "mà", "lại"}

POSITIVE_KEYWORDS = {"ngon", "đẹp", "tốt", "thích", "hài lòng", "tuyệt", "ưng", "nhanh", "nhiệt tình"}


def classify_error_cause(
    text: str,
    true_label: int,
    pred_label: int,
    source_domain: str,
    target_domain: str,
    rating: Optional[float] = None,
) -> tuple[str, str]:
    """Classify the likely root cause of a misprediction according to Section 16."""
    text_lower = text.lower()
    words = text_lower.split()

    # 1. Review quá ngắn
    if len(words) <= 4:
        return (
            "Review quá ngắn",
            f"Văn bản chỉ có {len(words)} từ, thiếu ngữ cảnh và tín hiệu ngữ nghĩa để mô hình phân loại chính xác."
        )

    # 2. Mâu thuẫn Rating và Nội dung
    if rating is not None:
        if rating >= 4 and true_label == 0:
            return (
                "Rating và nội dung không nhất quán",
                f"Đánh giá {rating} sao nhưng nội dung phàn nàn/tiêu cực gây nhiễu nhãn."
            )
        elif rating <= 2 and true_label == 2:
            return (
                "Rating và nội dung không nhất quán",
                f"Đánh giá {rating} sao nhưng nội dung khen ngợi/tích cực gây phân vân cho mô hình."
            )

    # 3. Phủ định (Negation)
    has_negation = any(k in text_lower for k in NEGATION_KEYWORDS)
    if has_negation:
        if (true_label == 0 and pred_label == 2) or (true_label == 2 and pred_label == 0):
            return (
                "Phủ định phức tạp",
                "Câu chứa từ phủ định làm đảo ngược ngữ nghĩa (ví dụ: 'không tệ', 'chẳng ngon lành gì') khiến mô hình bắt từ khóa độc lập bị nhầm."
            )

    # 4. Mixed Sentiment (Vừa khen vừa chê)
    has_pos = any(k in text_lower for k in POSITIVE_KEYWORDS)
    has_contrast = any(k in text_lower for k in CONTRAST_KEYWORDS)
    if has_pos and has_contrast:
        return (
            "Mixed Sentiment",
            "Đánh giá chứa cả mặt tích cực lẫn tiêu cực nối bởi từ đối lập ('nhưng', 'tuy nhiên') gây khó khăn cho việc gán 1 nhãn duy nhất."
        )

    # 5. Sarcasm (Mỉa mai / châm biếm)
    if true_label == 0 and any(k in text_lower for k in ["tuyệt vời", "quá đỉnh", "tốt ghê", "cảm ơn"]) and ("..." in text or "!" in text):
        return (
            "Sarcasm (Mỉa mai)",
            "Câu sử dụng từ ngữ tích cực bề mặt mang tính châm biếm/mỉa mai để thể hiện thái độ thất vọng."
        )

    # 6. Domain-specific vocabulary / Domain Shift
    if source_domain == "A" and target_domain == "B":
        # Model trained on Books (A), tested on Food (B)
        found_b = [k for k in DOMAIN_B_KEYWORDS if k in text_lower]
        if found_b:
            return (
                "Domain-specific vocabulary",
                f"Mô hình học từ Sách chưa từng gặp các khái niệm ẩm thực đặc thù: {', '.join(found_b[:3])}."
            )
    elif source_domain == "B" and target_domain == "A":
        # Model trained on Food (B), tested on Books (A)
        found_a = [k for k in DOMAIN_A_KEYWORDS if k in text_lower]
        if found_a:
            return (
                "Domain-specific vocabulary",
                f"Mô hình học từ Nhà hàng chưa từng gặp các khái niệm xuất bản/sách đặc thù: {', '.join(found_a[:3])}."
            )

    # 7. Domain Shift chung
    return (
        "Domain Shift",
        f"Cách hành văn và ngữ cảnh thể hiện cảm xúc khi chuyển từ miền {source_domain} sang miền {target_domain} có sự khác biệt lớn."
    )


def extract_error_cases(
    df: pd.DataFrame,
    source_domain: str,
    target_domain: str,
    model_name: str = "PhoBERT",
    text_col: str = "text_clean",
    true_label_col: str = "label",
    pred_label_col: str = "pred",
    max_cases_per_type: int = 3,
    total_cases: int = 15,
) -> pd.DataFrame:
    """Extract representative error cases categorized into error types."""
    errors = df[df[true_label_col] != df[pred_label_col]].copy()
    if errors.empty:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {}

    for _, row in errors.iterrows():
        text = str(row[text_col])
        true_l = int(row[true_label_col])
        pred_l = int(row[pred_label_col])
        rating = row.get("rating")

        error_type, reason = classify_error_cause(
            text=text,
            true_label=true_l,
            pred_label=pred_l,
            source_domain=source_domain,
            target_domain=target_domain,
            rating=rating,
        )

        current_count = type_counts.get(error_type, 0)
        if current_count < max_cases_per_type:
            type_counts[error_type] = current_count + 1
            results.append({
                "id": row.get("id", ""),
                "scenario": f"{source_domain} -> {target_domain}",
                "model": model_name,
                "review": text,
                "true_label": f"{true_l} ({LABEL_MAP.get(true_l, '')})",
                "pred_label": f"{pred_l} ({LABEL_MAP.get(pred_l, '')})",
                "error_type": error_type,
                "analysis": reason,
            })

        if len(results) >= total_cases:
            break

    return pd.DataFrame(results)


def format_error_table_markdown(df_errors: pd.DataFrame) -> str:
    """Format extracted errors into a clean Markdown table for project report."""
    if df_errors.empty:
        return "Không có trường hợp dự đoán sai."

    lines = [
        "| STT | Kịch bản | Nhãn thật | Dự đoán | Phân loại lỗi | Review | Phân tích nguyên nhân |",
        "| :---: | :---: | :---: | :---: | :--- | :--- | :--- |",
    ]

    for idx, row in df_errors.iterrows():
        rev = row['review'].replace("\n", " ").strip()
        if len(rev) > 75:
            rev = rev[:72] + "..."
        lines.append(
            f"| {idx + 1} | {row['scenario']} | {row['true_label']} | {row['pred_label']} | **{row['error_type']}** | {rev} | {row['analysis']} |"
        )

    return "\n".join(lines)
