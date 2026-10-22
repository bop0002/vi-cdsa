# Machine Learning Baseline

Phạm vi: phần Thành viên 2 trong kế hoạch nhóm. Hai mô hình là TF-IDF +
Logistic Regression và TF-IDF + LinearSVC. Dùng dữ liệu đã xử lý, không chạy
lại preprocessing hay thay đổi nhãn/split. Không triển khai PhoBERT hoặc EDA.

## Cài đặt và chạy

Chạy từ thư mục gốc repository, Python >= 3.10:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-baseline.txt
.\.venv\Scripts\python -m src.run_baseline all
```

Hoặc chạy từng giai đoạn:

```powershell
.\.venv\Scripts\python -m src.run_baseline train
.\.venv\Scripts\python -m src.run_baseline evaluate
.\.venv\Scripts\python -m src.run_baseline summarize
```

`evaluate` nạp checkpoint đã lưu, không train lại. `summarize` đọc metrics đã xuất.
Train/evaluate yêu cầu thư mục đích trống để không ghi đè kết quả.
Để chạy thí nghiệm mới:

```powershell
.\.venv\Scripts\python -m src.run_baseline all --config configs/baseline.json --model-dir models/baseline_v2 --output-dir results/baseline_v2
```

Có thể dùng `--data-dir` để chỉ định thư mục dữ liệu khác cùng schema.
Chỉ nạp checkpoint joblib do bạn hoặc nguồn tin cậy tạo.

## Quy trình thí nghiệm

- Đọc `data/processed/{A,B}_{train,val,test,test_balanced}.parquet`
  bằng `src.dataset.load_split`; dùng duy nhất `text_ml` làm đặc trưng.
- Giữ nguyên dữ liệu và nhãn 0=negative, 1=neutral, 2=positive.
- Fit TF-IDF trên train của miền nguồn. Validation/test chỉ được transform.
- Chọn C/class_weight bằng Macro-F1 trên validation **cùng miền nguồn**.
  Không dùng validation miền đích để chọn mô hình cross-domain.
- Nếu hòa điểm, giữ cấu hình xuất hiện đầu tiên trong danh sách tìm kiếm.
  Giữ mô hình đã fit trên train; không refit train+val.
- Với mỗi mô hình, train riêng trên A và B rồi đánh giá A→A, A→B, B→A, B→B.
  Thực hiện cho cả test gốc và test cân bằng: tổng 4 checkpoint, 16 dòng kết quả.
- Báo lỗi khi không hội tụ, dữ liệu sai schema/nhãn/rỗng, trùng ID hoặc
  trùng `text_clean` giữa các tập chính. Kiểm tra test cân bằng là tập con
  nhất quán với test gốc; không xem sự giao nhau này là rò rỉ.
- Khi evaluate, đối chiếu SHA-256 train/val và checkpoint với manifest.

## Cấu hình mặc định

`configs/baseline.json` chứa các lựa chọn triển khai có thể thay đổi:

| Tham số | Giá trị |
|---|---|
| Seed | 42 |
| TF-IDF n-gram | 1–2 |
| max_features | 20.000 |
| min_df | 2 |
| sublinear_tf | true |
| C | 0,1; 1; 10 |
| class_weight | null; balanced |
| max_iter | 5.000 |

Seed, n-gram và khoảng max_features bám kế hoạch; min_df, sublinear_tf,
lưới C/weight và max_iter là cấu hình khởi đầu do pipeline lựa chọn,
không phải thông số được khẳng định đã tối ưu trước khi thực nghiệm.
Tokenization dùng mặc định của TfidfVectorizer trên văn bản đã tách từ.
LR dùng lbfgs; LinearSVC dùng dual=True. ConvergenceWarning làm dừng chạy
để tránh âm thầm báo cáo một mô hình chưa hội tụ.

## Đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `models/baseline/*.joblib` | Toàn bộ TF-IDF + classifier, được gitignore |
| `models/baseline/manifest.json` | Cấu hình, phiên bản, SHA-256, thời gian, mọi thử nghiệm validation |
| `results/baseline/manifest.json` | Manifest kèm SHA-256 test |
| `results/baseline/metrics/summary.csv` | Accuracy, Macro precision/recall/F1, Weighted-F1, thời gian predict |
| `results/baseline/metrics/per_class.csv` | Precision, recall, F1, support từng lớp |
| `results/baseline/metrics/comparison_*.csv` | Bảng hai mô hình × bốn chiều |
| `results/baseline/metrics/performance_drop.csv` | F1 cùng miền trừ F1 khác miền, giữ miền train |
| `results/baseline/confusion_matrix/` | CSV và PNG; hàng=nhãn thật, cột=dự đoán |
| `results/baseline/predictions/` | ID, text_ml, nhãn thật/dự đoán để bàn giao phần phân tích lỗi |
| `results/baseline/figures/` | Biểu đồ so sánh Macro-F1 |
| `results/baseline/report.md` | Bảng kết quả tự sinh |

Điểm số ở thang 0–1; Performance Drop có thể âm nếu cross-domain tốt hơn.
Test gốc và test cân bằng cần đọc riêng vì phân bố nhãn khác nhau.
Không suy ra ý nghĩa thống kê từ chênh lệch điểm đơn lẻ: pipeline hiện dùng
một seed/split cố định, chưa tính khoảng tin cậy hay kiểm định thống kê.
Thời gian chạy phụ thuộc phần cứng; phiên bản thư viện được lưu trong manifest
để giúp tái hiện môi trường. Dữ liệu miền A chỉ là sách, miền B là nhà hàng;
không khái quát kết luận cho toàn bộ thương mại điện tử/khách sạn.

## Kiểm thử

```powershell
.\.venv\Scripts\python -m pytest tests/test_baseline.py -q
```

Kiểm thử dùng dữ liệu nhỏ riêng biệt: chạy end-to-end, kiểm tra vocabulary
không chứa token chỉ có ở validation/test, chọn đúng Macro-F1 validation,
đối chiếu metrics/confusion matrix với predictions, chặn dữ liệu thay đổi
và ghi đè đầu ra.
