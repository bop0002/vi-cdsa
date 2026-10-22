# So sánh dữ liệu và mô hình

Dữ liệu cũ: `b8ef6cf4a969a768798cd1e436abd11766ba3aee`.
Dữ liệu mới: `c809c2e1371eeb6fd85388f04be70e04d02a442d`.

Train mới từ đầu: TF-IDF và classifier mới hoàn toàn; không warm-start, không partial_fit.
Cùng cấu hình: True; cùng phiên bản thư viện: True.
Dùng train chuẩn cho cả hai phiên bản; train_balanced mới chỉ được thống kê, không dùng trong lần train này.

| Tệp | Số dòng cũ | Số dòng mới | ID thêm | ID bỏ | Nhãn đổi (ID chung) | Text ML đổi (ID chung) |
|---|---:|---:|---:|---:|---:|---:|
| A_test.parquet | 6288 | 6288 | 0 | 0 | 0 | 0 |
| A_test_balanced.parquet | 1017 | 1017 | 0 | 0 | 0 | 0 |
| A_train.parquet | 25485 | 30785 | 15557 | 10257 | 0 | 0 |
| A_train_balanced.parquet | 0 | 18471 | — | — | — | — |
| A_val.parquet | 6536 | 6536 | 0 | 0 | 0 | 0 |
| B_test.parquet | 3121 | 3121 | 0 | 0 | 0 | 0 |
| B_test_balanced.parquet | 816 | 816 | 0 | 0 | 0 | 0 |
| B_train.parquet | 25485 | 30785 | 10957 | 5657 | 0 | 0 |
| B_train_balanced.parquet | 0 | 18471 | — | — | — | — |
| B_val.parquet | 3152 | 3152 | 0 | 0 | 0 | 0 |

Các tập test giữ nguyên ID, text_ml và nhãn: True.
Đối chiếu theo ID, không phụ thuộc thứ tự dòng. SHA-256 kết quả đã được đối chiếu với snapshot Git.
File Parquet thay đổi byte không nhất thiết có nghĩa nội dung đầu vào mô hình thay đổi.

## Phân bố train

| Phiên bản | Miền | Negative | Neutral | Positive |
|---|---|---:|---:|---:|
| old | A | 5374 | 2141 | 17970 |
| old | B | 5374 | 2141 | 17970 |
| new | A | 12314 | 6157 | 12314 |
| new | B | 12314 | 6157 | 12314 |

## Nguồn train

| Phiên bản | Tệp | Nguồn | Số mẫu |
|---|---|---|---:|
| old | A_train.parquet | tiki_book | 25485 |
| new | A_train.parquet | tiki_book | 22781 |
| new | A_train.parquet | tiki_book_extra | 8004 |
| old | B_train.parquet | res1 | 25485 |
| new | B_train.parquet | res1 | 19828 |
| new | B_train.parquet | foody | 10957 |

## test

| Mô hình | Chiều | Macro-F1 cũ | Macro-F1 mới | Chênh lệch mới − cũ |
|---|---|---:|---:|---:|
| linear_svc | A→A | 0.6988 | 0.6963 | -0.0025 |
| linear_svc | A→B | 0.6942 | 0.6880 | -0.0062 |
| linear_svc | B→A | 0.5371 | 0.5123 | -0.0248 |
| linear_svc | B→B | 0.7635 | 0.7614 | -0.0021 |
| logistic_regression | A→A | 0.6845 | 0.6997 | +0.0152 |
| logistic_regression | A→B | 0.6763 | 0.6809 | +0.0046 |
| logistic_regression | B→A | 0.4989 | 0.5150 | +0.0161 |
| logistic_regression | B→B | 0.7602 | 0.7627 | +0.0025 |

## test_balanced

| Mô hình | Chiều | Macro-F1 cũ | Macro-F1 mới | Chênh lệch mới − cũ |
|---|---|---:|---:|---:|
| linear_svc | A→A | 0.6808 | 0.7013 | +0.0205 |
| linear_svc | A→B | 0.6684 | 0.6832 | +0.0148 |
| linear_svc | B→A | 0.5725 | 0.6092 | +0.0367 |
| linear_svc | B→B | 0.7504 | 0.7852 | +0.0348 |
| logistic_regression | A→A | 0.7302 | 0.7298 | -0.0003 |
| logistic_regression | A→B | 0.7043 | 0.6817 | -0.0227 |
| logistic_regression | B→A | 0.5973 | 0.5951 | -0.0023 |
| logistic_regression | B→B | 0.7940 | 0.7676 | -0.0263 |

## Giới hạn diễn giải

Đây là so sánh điểm số một seed, chưa phải kiểm định ý nghĩa thống kê.
Thay đổi đồng thời quy mô, nguồn và tỷ lệ nhãn train nên không quy chênh lệch cho riêng một yếu tố.
Tham số được chọn lại trên validation theo cùng lưới tìm kiếm.
Test cân bằng là tập con test gốc; không coi hai bảng là hai kiểm chứng độc lập.
