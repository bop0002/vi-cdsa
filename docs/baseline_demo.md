# Giao diện đánh giá bình luận

Giao diện desktop Tkinter dùng 4 checkpoint hiện có trong `models/baseline/`.
Không train lại mô hình. Chọn thuật toán và miền train tương ứng với checkpoint.

Chạy từ thư mục gốc repo (Windows, Java và Python có Tkinter):

```powershell
.\.venv\Scripts\python -m pip install -r requirements-demo.txt
.\.venv\Scripts\python -m src.baseline_demo --prepare
.\.venv\Scripts\python -m src.baseline_demo
```

`--prepare` tải tài nguyên VnCoreNLP vào `models/vncorenlp/` một lần.
Khi đã cài đủ tài nguyên, giao diện dự đoán tại máy; không gửi bình luận lên dịch vụ khác.
Nhập một dòng bình luận, chọn LinearSVC hoặc Logistic Regression, chọn miền
train A (sách Tiki) hoặc B (nhà hàng), rồi nhấn nút hoặc Enter.
Kết quả: 0 = Tiêu cực, 1 = Trung tính, 2 = Tích cực.

Tiền xử lý dùng lại `clean_text → segment → to_ml` như lúc tạo dữ liệu train.
Demo dùng cùng VnCoreNLP word segmenter. Với đường dẫn repo chứa dấu hoặc
khoảng trắng, tài nguyên được sao chép nguyên trạng vào thư mục TEMP không dấu
cho phiên chạy để tránh lỗi đường dẫn của Java; không sửa tài nguyên gốc.
Vì vậy câu chỉ chứa tiêu đề tự sinh như “Cực kỳ hài lòng.” có thể bị loại hết;
giao diện yêu cầu nhập nội dung cụ thể hơn. Đầu vào rỗng, hơn 5.000 ký tự hoặc
không có đặc trưng nào trong từ vựng mô hình sẽ được báo lỗi rõ ràng.
Không hiển thị phần trăm độ tin cậy vì LinearSVC chưa được hiệu chỉnh xác suất.
Checkpoint được kiểm tra SHA-256 với manifest trước khi nạp và được cache trong
phiên chạy. Khởi động lại giao diện nếu thay checkpoint. Chỉ dùng checkpoint tin cậy.
