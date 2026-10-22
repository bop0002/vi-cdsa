# Thí nghiệm theo phiên bản dữ liệu

## Các nhánh và đầu ra

| Phiên bản dữ liệu | Nhánh | Checkpoint local | Kết quả |
|---|---|---|---|
| b8ef6cf | feature/ml-baseline-dataset-b8ef6cf | models/baseline | results/baseline |
| c809c2e | experiment/ml-baseline-dataset-c809c2e | models/baseline_c809c2e | results/baseline_c809c2e |

Nhánh cũ tách từ main local b8ef6cf, commit pipeline/giao diện/kết quả là 75e3c49.
Nhánh mới tách từ origin/main c809c2e, nhận lại commit pipeline bằng cherry-pick
(8601966). Main local không bị sửa. Chưa push hai nhánh lên GitHub.
Checkpoint nằm trong models/ được gitignore nên không đi theo Git khi đổi nhánh;
cần giữ đúng đường dẫn phiên bản, không sao chép đè checkpoint cũ.
Giao diện demo hiện vẫn dùng models/baseline (phiên bản cũ).

## Lệnh đã chạy

```powershell
.\.venv\Scripts\python -m src.run_baseline all --model-dir models/baseline_c809c2e --output-dir results/baseline_c809c2e
.\.venv\Scripts\python -m src.compare_dataset_versions --old-ref b8ef6cf --new-ref c809c2e --old-results results/baseline --new-results results/baseline_c809c2e --output-dir results/comparison_b8ef6cf_c809c2e
```

Train tạo TF-IDF và estimator mới hoàn toàn cho mỗi lần thử tham số. Không gọi
joblib.load, warm_start hay partial_fit khi train. Evaluate chỉ nạp checkpoint
vừa sinh từ thư mục model-dir mới. Cấu hình, seed và phiên bản thư viện giống
lần cũ; tham số thắng được chọn lại trên validation cùng miền.

Dữ liệu mới bổ sung train_balanced nhưng lần so sánh này dùng train chuẩn như
lần cũ. Báo cáo so sánh nằm tại results/comparison_b8ef6cf_c809c2e/report.md;
các CSV bên cạnh ghi số ID thêm/bỏ, thay đổi nội dung/nhãn, phân bố lớp/nguồn,
chênh lệch metrics. Snapshot cũ được đọc trực tiếp từ Git, không tạo lại dữ liệu.

## Kết quả và diễn giải

Train mỗi miền tăng từ 25.485 lên 30.785 mẫu (+5.300). Đây không chỉ là thêm
dữ liệu: một số ID cũ bị thay thế do sampling. Tỷ lệ Negative/Neutral/Positive
chuyển từ khoảng 21,09%/8,40%/70,51% sang 40%/20%/40%.
Miền A có 8.004 mẫu tiki_book_extra; miền B có 10.957 mẫu foody.
Nhãn và text_ml của các ID còn chung không đổi. Validation/test giữ nguyên
ID, nội dung đầu vào và nhãn, dù một số file Parquet thay đổi byte.

Trên test gốc, Logistic Regression tăng Macro-F1 ở cả bốn chiều, LinearSVC
giảm ở cả bốn chiều. Trên test cân bằng, LinearSVC tăng cả bốn chiều còn LR
giảm cả bốn. Không thể kết luận dữ liệu mới luôn tốt hơn: kết quả phụ thuộc
mô hình và phân bố test. Chưa có kiểm định ý nghĩa thống kê; không quy toàn bộ
chênh lệch cho tăng số lượng mẫu vì nguồn và tỷ lệ nhãn cũng thay đổi.

## Kiểm chứng

15 tests baseline, inference, dataset và comparison đạt. Toàn bộ suite trước
khi thêm 2 tests comparison: 88 đạt, 2 lỗi ở tests/test_segment.py liên quan
Java/VnCoreNLP trên môi trường Windows hiện tại. Không sửa phần preprocessing
trong công việc này. Train/evaluate trên Parquet đã xử lý hoàn tất bình thường.

## Khi có phiên bản dữ liệu tiếp theo

Commit công việc hiện tại, fetch origin, tạo nhánh experiment/ml-baseline-dataset-<sha>
từ origin/main mới, đưa commit code baseline cần thiết sang nhánh đó. Train mới
với model-dir và output-dir riêng theo SHA; chạy compare_dataset_versions với
hai SHA và hai thư mục kết quả tương ứng. Không train vào thư mục checkpoint
cũ. Không dùng tập test để chọn cấu hình.
