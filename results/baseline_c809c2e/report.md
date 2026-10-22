# Kết quả Machine Learning Baseline

Macro-F1 là chỉ số chính. Các điểm số ở thang 0–1.
Tham số được chọn trên validation của miền train; TF-IDF chỉ fit trên train.
Performance Drop = F1 cùng miền − F1 khác miền, giữ nguyên miền train.
Test cân bằng là tập con của test gốc, không phải thí nghiệm độc lập.
Các kết quả là ước lượng điểm của một split/seed; chưa kiểm định ý nghĩa thống kê.

## test

| Model | A→A | B→B | A→B | B→A |
|---|---:|---:|---:|---:|
| linear_svc | 0.6963 | 0.7614 | 0.6880 | 0.5123 |
| logistic_regression | 0.6997 | 0.7627 | 0.6809 | 0.5150 |

## test_balanced

| Model | A→A | B→B | A→B | B→A |
|---|---:|---:|---:|---:|
| linear_svc | 0.7013 | 0.7852 | 0.6832 | 0.6092 |
| logistic_regression | 0.7298 | 0.7676 | 0.6817 | 0.5951 |

## Performance Drop

| Model | Miền train | Tập test | Drop Macro-F1 |
|---|---|---|---:|
| linear_svc | A | test | 0.0083 |
| linear_svc | A | test_balanced | 0.0182 |
| linear_svc | B | test | 0.2491 |
| linear_svc | B | test_balanced | 0.1760 |
| logistic_regression | A | test | 0.0189 |
| logistic_regression | A | test_balanced | 0.0482 |
| logistic_regression | B | test | 0.2477 |
| logistic_regression | B | test_balanced | 0.1726 |

