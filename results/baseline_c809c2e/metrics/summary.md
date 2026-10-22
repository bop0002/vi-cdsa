# Kết quả đánh giá mô hình

Bộ kết quả: `baseline_c809c2e` · Nguồn: [summary.csv](summary.csv).

A = đánh giá sách Tiki; B = đánh giá nhà hàng. Chiều đọc: **miền train → miền test**.
A→A, B→B: cùng miền. A→B, B→A: khác miền.

Điểm số hiển thị trên thang **0–100%**, làm tròn 2 chữ số thập phân.
**Accuracy** là tỷ lệ dự đoán đúng. **Macro-F1** là trung bình F1 của ba lớp, không phải tỷ lệ dự đoán đúng.
CSV gốc giữ nguyên độ chính xác số liệu. Không train hoặc đánh giá lại để tạo bảng này.

## Test gốc

### So sánh nhanh: Accuracy và Macro-F1

| Train → Test | Số mẫu test | LR · Accuracy | SVC · Accuracy | LR · Macro-F1 | SVC · Macro-F1 |
|---|---:|---:|---:|---:|---:|
| A→A | 6,288 | 86.61% | 87.26% | 69.97% | 69.63% |
| B→B | 3,121 | 88.31% | 87.06% | 76.27% | 76.14% |
| A→B | 3,121 | 81.67% | 82.60% | 68.09% | 68.80% |
| B→A | 6,288 | 68.07% | 66.63% | 51.50% | 51.23% |

### Logistic Regression · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 66.46% | 75.35% | 87.59% | 0.305 giây |
| B→B | 75.51% | 77.16% | 88.57% | 0.479 giây |
| A→B | 66.42% | 70.37% | 82.35% | 0.427 giây |
| B→A | 50.97% | 62.52% | 72.00% | 0.506 giây |

### LinearSVC · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 66.94% | 73.56% | 87.82% | 0.618 giây |
| B→B | 74.91% | 78.72% | 87.95% | 0.419 giây |
| A→B | 67.37% | 70.67% | 83.08% | 0.351 giây |
| B→A | 50.69% | 63.35% | 70.90% | 0.552 giây |

## Test cân bằng

### So sánh nhanh: Accuracy và Macro-F1

| Train → Test | Số mẫu test | LR · Accuracy | SVC · Accuracy | LR · Macro-F1 | SVC · Macro-F1 |
|---|---:|---:|---:|---:|---:|
| A→A | 1,017 | 74.14% | 72.07% | 72.98% | 70.13% |
| B→B | 816 | 78.06% | 79.04% | 76.76% | 78.52% |
| A→B | 816 | 69.61% | 69.98% | 68.17% | 68.32% |
| B→A | 1,017 | 61.46% | 62.14% | 59.51% | 60.92% |

### Logistic Regression · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 74.94% | 74.14% | 72.98% | 0.117 giây |
| B→B | 78.75% | 78.06% | 76.76% | 0.098 giây |
| A→B | 69.61% | 69.61% | 68.17% | 0.175 giây |
| B→A | 65.80% | 61.46% | 59.51% | 0.098 giây |

### LinearSVC · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 73.60% | 72.07% | 70.13% | 0.089 giây |
| B→B | 79.06% | 79.04% | 78.52% | 0.122 giây |
| A→B | 70.18% | 69.98% | 68.32% | 0.136 giây |
| B→A | 65.79% | 62.14% | 60.92% | 0.102 giây |

## Cách đọc các chỉ số

- **Precision (macro):** precision trung bình của ba lớp, mỗi lớp có trọng số bằng nhau.
- **Recall (macro):** recall trung bình của ba lớp, mỗi lớp có trọng số bằng nhau.
- **F1 (weighted):** F1 trung bình có trọng số theo số mẫu thật của từng lớp.
- **Test cân bằng:** tập con của test gốc với số mẫu bằng nhau giữa ba lớp; đọc riêng hai bảng.
- Điểm từng lớp nằm trong [per_class.csv](per_class.csv). Chênh lệch điểm chưa phải kết luận về ý nghĩa thống kê.
