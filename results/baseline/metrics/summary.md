# Kết quả đánh giá mô hình

Bộ kết quả: `baseline` · Nguồn: [summary.csv](summary.csv).

A = đánh giá sách Tiki; B = đánh giá nhà hàng. Chiều đọc: **miền train → miền test**.
A→A, B→B: cùng miền. A→B, B→A: khác miền.

Điểm số hiển thị trên thang **0–100%**, làm tròn 2 chữ số thập phân.
**Accuracy** là tỷ lệ dự đoán đúng. **Macro-F1** là trung bình F1 của ba lớp, không phải tỷ lệ dự đoán đúng.
CSV gốc giữ nguyên độ chính xác số liệu. Không train hoặc đánh giá lại để tạo bảng này.

## Test gốc

### So sánh nhanh: Accuracy và Macro-F1

| Train → Test | Số mẫu test | LR · Accuracy | SVC · Accuracy | LR · Macro-F1 | SVC · Macro-F1 |
|---|---:|---:|---:|---:|---:|
| A→A | 6,288 | 84.97% | 88.30% | 68.45% | 69.88% |
| B→B | 3,121 | 86.38% | 88.79% | 76.02% | 76.35% |
| A→B | 3,121 | 80.52% | 84.40% | 67.63% | 69.42% |
| B→A | 6,288 | 65.76% | 73.49% | 49.89% | 53.71% |

### Logistic Regression · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 65.43% | 75.14% | 86.68% | 0.273 giây |
| B→B | 74.42% | 79.87% | 87.59% | 0.255 giây |
| A→B | 65.94% | 70.87% | 81.90% | 0.210 giây |
| B→A | 49.74% | 61.94% | 70.29% | 0.273 giây |

### LinearSVC · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 68.62% | 71.31% | 88.59% | 0.304 giây |
| B→B | 76.64% | 76.07% | 88.75% | 0.266 giây |
| A→B | 70.44% | 68.53% | 84.20% | 0.223 giây |
| B→A | 52.73% | 61.75% | 76.26% | 0.254 giây |

## Test cân bằng

### So sánh nhanh: Accuracy và Macro-F1

| Train → Test | Số mẫu test | LR · Accuracy | SVC · Accuracy | LR · Macro-F1 | SVC · Macro-F1 |
|---|---:|---:|---:|---:|---:|
| A→A | 1,017 | 73.55% | 69.81% | 73.02% | 68.08% |
| B→B | 816 | 79.66% | 76.47% | 79.40% | 75.04% |
| A→B | 816 | 71.32% | 68.63% | 70.43% | 66.84% |
| B→A | 1,017 | 61.16% | 60.47% | 59.73% | 57.25% |

### Logistic Regression · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 73.54% | 73.55% | 73.02% | 0.057 giây |
| B→B | 79.43% | 79.66% | 79.40% | 0.120 giây |
| A→B | 70.83% | 71.32% | 70.43% | 0.067 giây |
| B→A | 64.67% | 61.16% | 59.73% | 0.049 giây |

### LinearSVC · Chi tiết

| Train → Test | Precision (macro) | Recall (macro) | F1 (weighted) | Thời gian dự đoán cả tập |
|---|---:|---:|---:|---:|
| A→A | 71.63% | 69.81% | 68.08% | 0.049 giây |
| B→B | 77.13% | 76.47% | 75.04% | 0.072 giây |
| A→B | 70.47% | 68.63% | 66.84% | 0.067 giây |
| B→A | 63.57% | 60.47% | 57.25% | 0.043 giây |

## Cách đọc các chỉ số

- **Precision (macro):** precision trung bình của ba lớp, mỗi lớp có trọng số bằng nhau.
- **Recall (macro):** recall trung bình của ba lớp, mỗi lớp có trọng số bằng nhau.
- **F1 (weighted):** F1 trung bình có trọng số theo số mẫu thật của từng lớp.
- **Test cân bằng:** tập con của test gốc với số mẫu bằng nhau giữa ba lớp; đọc riêng hai bảng.
- Điểm từng lớp nằm trong [per_class.csv](per_class.csv). Chênh lệch điểm chưa phải kết luận về ý nghĩa thống kê.
