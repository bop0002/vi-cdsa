# Kết quả Machine Learning Baseline

Macro-F1 là chỉ số chính. Các điểm số ở thang 0–1.
Tham số được chọn trên validation của miền train; TF-IDF chỉ fit trên train.
Performance Drop = F1 cùng miền − F1 khác miền, giữ nguyên miền train.
Test cân bằng là tập con của test gốc, không phải thí nghiệm độc lập.
Các kết quả là ước lượng điểm của một split/seed; chưa kiểm định ý nghĩa thống kê.

## test

| Model | A→A | B→B | A→B | B→A |
|---|---:|---:|---:|---:|
| linear_svc | 0.6988 | 0.7635 | 0.6942 | 0.5371 |
| logistic_regression | 0.6845 | 0.7602 | 0.6763 | 0.4989 |

## test_balanced

| Model | A→A | B→B | A→B | B→A |
|---|---:|---:|---:|---:|
| linear_svc | 0.6808 | 0.7504 | 0.6684 | 0.5725 |
| logistic_regression | 0.7302 | 0.7940 | 0.7043 | 0.5973 |

## Performance Drop

| Model | Miền train | Tập test | Drop Macro-F1 |
|---|---|---|---:|
| linear_svc | A | test | 0.0046 |
| linear_svc | A | test_balanced | 0.0124 |
| linear_svc | B | test | 0.2264 |
| linear_svc | B | test_balanced | 0.1780 |
| logistic_regression | A | test | 0.0082 |
| logistic_regression | A | test_balanced | 0.0258 |
| logistic_regression | B | test | 0.2613 |
| logistic_regression | B | test_balanced | 0.1966 |

