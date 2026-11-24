# Phân Tích Cảm Xúc Đánh Giá Tiếng Việt Theo Hướng Cross-Domain (`vi-cdsa`)

Dự án nghiên cứu và xây dựng hệ thống phân loại cảm xúc văn bản tiếng Việt thành 3 lớp (**Negative**, **Neutral**, **Positive**) và đánh giá hiện tượng dịch chuyển miền dữ liệu (**Domain Shift**) khi chuyển giao mô hình giữa hai lĩnh vực khác biệt:

* **Miền A (Domain A - E-commerce):** Đánh giá sách trên sàn thương mại điện tử (**Tiki Books**).
* **Miền B (Domain B - F&B / Service):** Đánh giá nhà hàng, quán ăn (**Foody / Res1**).

---

## 1. Mục tiêu & Điểm nổi bật

* **Dữ liệu chuẩn hóa & Khử nhiễu nhãn (Gray-Zone Relabeling):** Sử dụng quy trình gán nhãn mù theo ngữ nghĩa văn bản cho hơn 52.000 mẫu "vùng xám" (rating mâu thuẫn với nội dung), được kiểm toán chéo với chuyên gia con người đạt độ đồng thuận **92.0%** (Miền A) và **82.0%** (Miền B).
* **Tập trung vào Phân bố Tự nhiên (Natural Distribution):** Giữ nguyên phân bố thực tế (79.510 mẫu/miền) và giải quyết mất cân bằng lớp (lớp Neutral chỉ chiếm ~8.25%) bằng hàm mất mát **Focal Loss** thay vì cắt bỏ dữ liệu.
* **Thực nghiệm In-Domain & Cross-Domain:** Đánh giá đầy đủ 4 kịch bản ($A \to A$, $B \to B$, $A \to B$, $B \to A$) bằng chỉ số chính **Macro-F1** và đo lường mức độ suy giảm hiệu năng (**Performance Drop**).
* **Phân tích lỗi chuyên sâu (Error Analysis):** Tự động trích xuất và phân loại các trường hợp dự đoán sai theo 8 nhóm nguyên nhân ngôn ngữ học (từ vựng chuyên ngành, phủ định phức tạp, nghịch ngữ/sarcasm, xung đột khía cạnh, review ngắn...).

---

## 2. Cấu trúc thư mục dự án

```text
vi-cdsa/
├── data/
│   ├── processed/
│   │   ├── full/                  # Dữ liệu sạch phân bố tự nhiên (*.parquet) và stats.json
│   │   ├── label_audit_A.csv      # 300 mẫu kiểm toán thủ công Miền A
│   │   └── label_audit_B.csv      # 300 mẫu kiểm toán thủ công Miền B
│   └── relabel/                   # Batch gán nhãn mù, kết quả AI và relabel_labels.csv
├── docs/
│   ├── PROJECT_SPEC.md            # Đặc tả kỹ thuật hệ thống chi tiết
│   ├── relabel_prompt.md          # Prompt & tiêu chí gán nhãn vùng xám
│   └── Kế Hoạch & Phân Chia Nhiệm Vụ Đề Tài Phân Tích Cảm Xúc.pdf
├── notebooks/
│   └── 03_phobert_colab.ipynb     # Notebook chạy PhoBERT 1-click trên Google Colab GPU T4
├── results/
│   ├── figures/                   # Biểu đồ Confusion Matrix (PNG)
│   └── metrics/                   # Kết quả thực nghiệm (CSV, JSON) & bảng Error Analysis (MD, CSV)
├── scripts/
│   ├── crawl_foody.py             # Script thu thập dữ liệu Foody
│   ├── crawl_tiki_books.py        # Script thu thập dữ liệu Tiki Books
│   ├── run_relabel_api.py         # Runner gọi API LLM gán nhãn vùng xám
│   ├── run_phobert.py             # Script CLI huấn luyện & đánh giá PhoBERT
│   └── generate_colab_notebook.py # Script tạo notebook Colab
├── src/
│   ├── config.py                  # Cấu hình đường dẫn, random seed (42), ngưỡng lọc
│   ├── text_norm.py               # Chuẩn hóa Unicode NFKC, teencode, giá tiền, emoji
│   ├── filters.py                 # Lọc độ dài (>=3 từ), ký tự tiếng Việt, deduplication
│   ├── segment.py                 # Tách từ tiếng Việt bằng VnCoreNLP (RDRSegmenter)
│   ├── io_sources.py              # Đọc và chuẩn hóa các nguồn dữ liệu thô
│   ├── prepare.py                 # Pipeline làm sạch dùng chung
│   ├── relabel.py                 # Export, validate, merge và QA kết quả gán nhãn
│   ├── split.py                   # Chia tập Stratified Group Split & ghép cặp 2 miền
│   ├── build_dataset.py           # Pipeline tổng hợp tạo tập dữ liệu sạch (.parquet)
│   ├── dataset.py                 # Helper load_split() nạp dữ liệu đã xử lý
│   ├── checks.py                  # Bộ kiểm tra tính toàn vẹn dữ liệu sau build
│   ├── focal_loss.py              # Cài đặt Multi-class Focal Loss
│   ├── phobert.py                 # Pipeline PhoBERT (Dataset, Dynamic Padding, Trainer, Evaluator)
│   └── error_analysis.py          # Module trích xuất & phân loại lỗi Cross-Domain
├── tests/                         # Bộ kiểm thử tự động (pytest)
├── .env.example                   # Mẫu biến môi trường cho Relabel API
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

## 3. Thông tin Bộ dữ liệu (`data/processed/full/`)

Dữ liệu đã được xử lý sẵn và lưu dưới dạng `.parquet` với 2 biến thể cột văn bản chính: `text_ml` (cho TF-IDF) và `text_dl` (đã tách từ `_` bởi VnCoreNLP cho PhoBERT).

| Tập dữ liệu | Số lượng mẫu (Miền A = Miền B) | Negative (`0`) | Neutral (`1`) | Positive (`2`) | Độ dài TB (Miền A / B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (80%)** | **63.610** | 15.769 (24.8%) | 5.251 (8.3%) | 42.590 (67.0%) | 33.5 từ / 64.8 từ |
| **Validation (10%)** | **7.950** | 1.971 (24.8%) | 656 (8.3%) | 5.323 (67.0%) | 31.4 từ / 68.4 từ |
| **Test (10%)** | **7.950** | 1.971 (24.8%) | 656 (8.3%) | 5.323 (67.0%) | 33.2 từ / 61.2 từ |
| **Tổng mỗi miền** | **79.510** | **19.711** | **6.563** | **53.236** | — |

---

## 4. Cài đặt môi trường

Yêu cầu **Python >= 3.10**.

```bash
# 1. Clone repository
git clone https://github.com/bop0002/vi-cdsa.git
cd vi-cdsa

# 2. Tạo và kích hoạt môi trường ảo (khuyên dùng)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

*(Tùy chọn)* Nếu muốn chạy lại script gán nhãn qua API (`scripts/run_relabel_api.py`), sao chép file `.env.example` thành `.env` và điền thông tin API key của bạn:
```bash
cp .env.example .env
```

---

## 5. Hướng dẫn sử dụng

### Bước 1: Chuẩn bị & Tiền xử lý dữ liệu (Tùy chọn)
Các file `.parquet` sạch đã được đóng gói sẵn trong `data/processed/full/` nên bạn có thể bỏ qua bước này để chạy mô hình ngay. Nếu muốn build lại từ dữ liệu thô:
```bash
python -m src.build_dataset --variant full
```

### Bước 2: Huấn luyện & Đánh giá PhoBERT (`vinai/phobert-base-v2`)

* **Cách A — Chạy trên Google Colab (Khuyên dùng, dùng GPU T4 miễn phí):**
  Mở trực tiếp notebook [`notebooks/03_phobert_colab.ipynb`](notebooks/03_phobert_colab.ipynb), chọn **Runtime $\to$ Change runtime type $\to$ T4 GPU** và chọn **Run all**. Notebook sẽ tự động kéo code/dữ liệu, huấn luyện cả Cross-Entropy và Focal Loss, đánh giá 4 kịch bản và nén kết quả tải về máy.

* **Cách B — Chạy qua dòng lệnh (CLI) trên máy có GPU:**
  ```bash
  python scripts/run_phobert.py --epochs 3 --batch_size 32 --max_length 128
  ```
  Để chạy thử nghiệm nhanh (dry-run) trên tập mẫu nhỏ:
  ```bash
  python scripts/run_phobert.py --epochs 1 --batch_size 16 --sample_limit 200
  ```

### Bước 3: Chạy kiểm thử (Unit Tests)
```bash
# Kiểm thử pipeline PhoBERT, Focal Loss, Error Analysis và Relabel API
python -m pytest tests/test_focal_loss.py tests/test_phobert_pipeline.py tests/test_error_analysis.py tests/test_run_relabel_api.py

# Hoặc chạy toàn bộ test suite (cần cài đủ requirements.txt)
python -m pytest tests/
```

*(Lưu ý: Phần mô hình Baseline truyền thống TF-IDF + Logistic Regression / LinearSVC đang được phát triển độc lập theo phân công Thành viên 2).*

---

## 6. Kết quả thực nghiệm (PhoBERT trên dữ liệu tự nhiên)

| Kịch bản | Loại | Hàm Loss | Macro-F1 | Accuracy | F1 (Negative) | F1 (Neutral) | F1 (Positive) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A $\to$ A** | In-domain | Cross-Entropy | 0.8047 | 90.50% | 0.9129 | 0.5546 | 0.9465 |
| **A $\to$ A** | In-domain | **Focal Loss** | **0.8109** | 89.16% | 0.9122 | **0.5850** | 0.9355 |
| **B $\to$ B** | In-domain | Cross-Entropy | 0.8536 | 93.23% | 0.9445 | 0.6529 | 0.9635 |
| **B $\to$ B** | In-domain | **Focal Loss** | **0.8588** | 92.72% | 0.9431 | **0.6734** | 0.9601 |
| **A $\to$ B** | Cross-domain | Cross-Entropy | 0.7869 | 91.12% | 0.9150 | 0.4953 | 0.9503 |
| **A $\to$ B** | Cross-domain | **Focal Loss** | **0.8211** | 91.23% | 0.9126 | **0.5969** | 0.9538 |
| **B $\to$ A** | Cross-domain | Cross-Entropy | **0.7679** | 87.26% | 0.8710 | **0.5042** | 0.9285 |
| **B $\to$ A** | Cross-domain | Focal Loss | 0.7596 | 85.02% | 0.8714 | 0.4984 | 0.9091 |

### Các phát hiện chính:
1. **Hiệu quả của Focal Loss:** Cải thiện rõ rệt khả năng nhận diện lớp thiểu số **Neutral**, đặc biệt tăng **+10.16%** điểm F1-Neutral ở kịch bản Cross-domain $A \to B$ (`0.4953` $\to$ `0.5969`).
2. **Bất đối xứng chuyển miền (Domain Asymmetry):** Chiều $A \to B$ (Sách $\to$ Nhà hàng) chuyển giao rất tốt, trong khi chiều $B \to A$ (Nhà hàng $\to$ Sách) bị sụt giảm ~8.6%–9.9% do chênh lệch độ dài câu (65 từ vs 33 từ) và từ vựng xuất bản/vận chuyển đặc thù (*màng co, cấn gáy, bookmark, shipper*).

Chi tiết bảng phân tích lỗi và hình ảnh ma trận nhầm lẫn xem tại [`results/metrics/error_analysis.md`](results/metrics/error_analysis.md) và [`results/figures/phobert_confusion_matrices.png`](results/figures/phobert_confusion_matrices.png).

---

## 7. Tài liệu & Giấy phép

* **Đặc tả hệ thống:** [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md)
* **Giấy phép:** Dự án được phân phối theo giấy phép MIT — xem chi tiết tại [`LICENSE`](LICENSE).
