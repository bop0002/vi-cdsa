# ĐẶC TẢ KỸ THUẬT HỆ THỐNG (SYSTEM SPECIFICATION)
## DỰ ÁN: PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ TIẾNG VIỆT THEO HƯỚNG CROSS-DOMAIN (VI-CDSA)

---

## 1. TỔNG QUAN HỆ THỐNG & MỤC TIÊU NGHIÊN CỨU

Hệ thống được thiết kế để phân loại cảm xúc văn bản tiếng Việt thành 3 lớp (`0: Negative`, `1: Neutral`, `2: Positive`) và đánh giá hiện tượng trôi dạt miền dữ liệu (**Domain Shift**) khi chuyển giao mô hình giữa hai lĩnh vực có đặc thù ngôn ngữ khác biệt:
- **Miền A (Source/Target):** Đánh giá Sách trên sàn Thương mại Điện tử (Tiki Books).
- **Miền B (Source/Target):** Đánh giá Dịch vụ Ăn uống / Nhà hàng (Foody & Res1).

---

## 2. KIẾN TRÚC MODULE HỆ THỐNG THỰC TẾ

```
vi-cdsa/
├── src/
│   ├── config.py            # Hằng số cấu hình đường dẫn, seed (42), ngưỡng lọc
│   ├── text_norm.py         # Chuẩn hóa Unicode NFKC, teencode, regex giá tiền, emoji
│   ├── filters.py           # Lọc rác (>=3 từ, ký tự tiếng Việt), deduplication
│   ├── segment.py           # Tách từ tiếng Việt qua VnCoreNLP (RDRSegmenter)
│   ├── relabel.py           # Engine gán lại nhãn vùng xám & thẩm định chất lượng AI
│   ├── split.py             # Chia tập Stratified Group Split & căn bằng quy mô 2 miền
│   ├── build_dataset.py     # Pipeline hợp nhất tạo tập dữ liệu sạch (natural / balanced)
│   ├── focal_loss.py        # Hàm mất mát Focal Loss cho dữ liệu mất cân bằng nhãn
│   ├── phobert.py           # Backbone vinai/phobert-base-v2, Dynamic Padding, Trainer
│   └── error_analysis.py    # Bộ lọc và phân loại 8 nhóm lỗi Cross-domain thực tế
├── scripts/
│   ├── run_relabel_api.py   # Runner gọi API LLM (temperature=0) gán nhãn mù batch
│   ├── run_phobert.py       # CLI chạy huấn luyện & đánh giá PhoBERT
│   └── generate_colab_notebook.py
├── notebooks/
│   └── 03_phobert_colab.ipynb # Notebook thực nghiệm 1-click trên Google Colab GPU T4
├── data/
│   ├── processed/full/      # Dữ liệu sạch phân bố tự nhiên (Parquet: 79.510 mẫu/miền)
│   └── relabel/             # Batches (input mù), Results (AI output), relabel_labels.csv
└── results/
    ├── metrics/             # Kết quả Macro-F1, Accuracy, Error Analysis (CSV, JSON, MD)
    └── figures/             # Confusion Matrix 4 kịch bản (PNG)
```

---

## 3. QUY TRÌNH TIỀN XỬ LÝ & SCHEMA DỮ LIỆU SẠCH

### 3.1. Schema chuẩn của mỗi mẫu dữ liệu (`*.parquet`)
| Trường dữ liệu | Kiểu | Mô tả chi tiết |
|---|---|---|
| `id` | `str` | Mã định danh duy nhất (`tiki_<id>` hoặc `res1_<id>`) |
| `domain` | `str` | `A` (Sách) hoặc `B` (Nhà hàng) |
| `source` | `str` | Nguồn thu thập (`tiki_book`, `tiki_book_extra`, `res1`, `foody`) |
| `group_id` | `str` | Nhóm sản phẩm/địa điểm dùng để chia tập chống rò rỉ (leakage) |
| `rating` | `float` | Điểm đánh giá gốc (1-5 sao hoặc 1-10 điểm) |
| `label` | `int` | Nhãn cảm xúc chuẩn hóa: `0: Negative`, `1: Neutral`, `2: Positive` |
| `label_source` | `str` | Nguồn gán nhãn: `rating` (từ sao), `dataset` (gốc), `content` (AI gán lại theo nội dung) |
| `split` | `str` | Phân vùng dữ liệu: `train` / `val` / `test` |
| `text_raw` | `str` | Văn bản gốc nguyên bản |
| `text_clean` | `str` | Văn bản đã chuẩn hóa Unicode NFKC, lọc emoji, xử lý teencode |
| `text_ml` | `str` | Định dạng cho mô hình Baseline TF-IDF (bỏ dấu phẩy) |
| `text_dl` | `str` | Định dạng cho PhoBERT (tách từ `_`, giữ dấu phẩy làm ngắt câu) |
| `n_words` | `int` | Số lượng từ trong câu sạch |

### 3.2. Tiêu chuẩn tiền xử lý văn bản
1. **Cắt tiêu đề tự sinh:** Loại bỏ các cụm từ do hệ thống Tiki tự thêm vào đầu câu dựa trên số sao (*"Cực kì hài lòng.", "Bình thường."*) để tránh rò rỉ nhãn vào nội dung.
2. **Quy tắc độ dài & ngôn ngữ:** Bắt buộc $\ge 3$ từ và chứa ít nhất 1 ký tự tiếng Việt có dấu.
3. **Chuẩn hóa giá tiền & teencode:** 
   - `kg` $\to$ `không` (trừ khi đứng sau chữ số mô tả khối lượng như `2kg`).
   - `k` sau số trước dấu phẩy giữ nguyên là đơn vị tiền tệ (`50 k,`), các vị trí khác đổi thành `không`.
4. **Tách từ tiếng Việt:** Sử dụng VnCoreNLP để liên kết từ ghép (`chất_lượng`, `phục_vụ`).

---

## 4. ĐẶC TẢ GÁN LẠI NHÃN "VÙNG XÁM" (CONTENT-BASED RELABELING)

### 4.1. Lý do cần gán nhãn lại theo nội dung
Kiểm tra nhãn thủ công D3 trên 600 mẫu ngẫu nhiên (`label_audit_A.csv`, `label_audit_B.csv`) cho thấy nhãn suy ra từ số sao có độ tin cậy rất thấp ở lớp giữa:
- **Miền A (Sách):** Đánh giá 3 sao thực chất có tới **71% là CHÊ** (Negative); chỉ **29% là Neutral thật**. Đánh giá 4 sao chứa phần lớn Neutral thật (khen chê lẫn lộn).
- **Miền B (Nhà hàng):** Nhãn Neutral có sẵn thực chất có tới **42% là KHEN CÓ CHÊ NHẸ** (Positive); chỉ **58% là Neutral thật**.
- *Hệ quả:* Khái niệm "Neutral" giữa hai miền bị ngược nhau hoàn toàn nếu chỉ dùng nhãn gốc, làm sai lệch bản chất của thí nghiệm Cross-Domain.

### 4.2. Quy chuẩn kiểm tra và thẩm định chất lượng AI gán nhãn
Hệ thống thiết lập **5 quy chuẩn kiểm định nghiêm ngặt** để xác nhận AI gán nhãn chính xác:

1. **Quy chuẩn Gán nhãn Mù (Blind Annotation):**
   - Toàn bộ 52.537 dòng thuộc "vùng xám" được trích xuất thành các batch JSONL chỉ gồm `{"id", "text"}`.
   - **Tuyệt đối không gửi số sao rating hoặc nhãn cũ** sang LLM để tránh mô hình bị thiên kiến mỏ neo (Anchor Bias).
2. **Quy chuẩn Ràng buộc Prompt & Heuristics:**
   - Sử dụng [`docs/relabel_prompt.md`](relabel_prompt.md) với định nghĩa ranh giới nhãn rõ ràng:
     - `0 (Negative)`: Chủ yếu chê/phàn nàn (kể cả chỉ chê đóng gói, shipper, sách rách, phục vụ kém).
     - `1 (Neutral)`: Khen chê cân bằng (*"ổn nhưng..."*), cảm xúc nhạt (*"bình thường, cũng được"*), hoặc chưa trải nghiệm (*"chưa đọc nên chưa biết"*).
     - `2 (Positive)`: Chủ yếu khen, hài lòng (khen lớn, chê nhỏ nhặt kết luận vẫn tốt).
   - Nhiệt độ lấy mẫu cố định `temperature = 0` để đảm bảo tính tái lập 100%.
   - Bắt buộc LLM phải đưa ra lý do ngắn bằng tiếng Việt (`note` tối đa 20 từ) để phục vụ kiểm toán giải thích.
3. **Quy chuẩn Độ đồng thuận với Kiểm toán Con người (D3 Audit Agreement Rate):**
   - So sánh nhãn do AI gán với 600 dòng đã được chuyên gia con người thẩm định thủ công:
     - Nếu chuyên gia xác nhận nhãn cũ đúng $\to$ AI bắt buộc phải giữ nguyên nhãn cũ.
     - Nếu chuyên gia xác nhận nhãn cũ sai $\to$ AI bắt buộc phải chuyển sang nhãn mới.
   - **Kết quả thực tế đạt được:**
     - **Miền A:** Đạt độ đồng thuận **92.0%** (103 / 112 mẫu).
     - **Miền B:** Đạt độ đồng thuận **82.0%** (82 / 100 mẫu).
     - *(Cả hai miền đều vượt ngưỡng mục tiêu tối thiểu $\ge 80\%$)*.
4. **Quy chuẩn Kiểm tra Tính toàn vẹn Cú pháp (Integrity Validation):**
   - Hàm `validate_results` trong [`src/relabel.py`](../src/relabel.py) kiểm tra:
     - 100% ID khớp chính xác với tập đầu vào, không có ID lạ, không trùng lặp.
     - Nhãn đầu ra bắt buộc $\in \{0, 1, 2\}$.
     - Cột `note` không được để trống.
5. **Quy chuẩn Lọc bỏ Thư rác (Spam Filtering):**
   - Tự động phát hiện và loại bỏ các đánh giá quảng cáo/spam do annotator đánh dấu `(spam)` ra khỏi tập dữ liệu huấn luyện.

---

## 5. THIẾT KẾ PHÂN CHIA DỮ LIỆU & CHIẾN LƯỢC MẤT CÂN BẰNG

### 5.1. Lựa chọn Phân bố Tự nhiên (Natural Distribution)
Thay vì dùng kỹ thuật Undersampling làm mất mát tới 75% dữ liệu (chỉ còn 15.000 mẫu), đề tài chọn huấn luyện trên **phân bố tự nhiên hoàn chỉnh** (79.510 mẫu/miền):
- **Train (80%):** 63.610 mẫu (Neg: 15.769, Neu: 5.251, Pos: 42.590).
- **Val (10%):** 7.950 mẫu (Neg: 1.971, Neu: 656, Pos: 5.323).
- **Test (10%):** 7.950 mẫu (Neg: 1.971, Neu: 656, Pos: 5.323).

### 5.2. Hàm mất mát Focal Loss giải quyết mất cân bằng
Để xử lý lớp Neutral chỉ chiếm ~8.2%, hệ thống triển khai Focal Loss:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad \text{với } \gamma = 2.0$$
Trọng số $\alpha_t$ được tính nghịch đảo theo tần suất xuất hiện của lớp trên tập huấn luyện, giúp triệt tiêu tổn thất từ các mẫu Positive quá dễ và tập trung cập nhật gradient cho lớp Neutral.

---

## 6. MA TRẬN THỰC NGHIỆM & PHÂN TÍCH HIỆN TƯỢNG DOMAIN SHIFT

### 6.1. Ma trận 4 Kịch bản Đánh giá
1. **$A \to A$ (In-domain):** Huấn luyện trên Tiki Sách $\to$ Kiểm thử trên Tiki Sách.
2. **$B \to B$ (In-domain):** Huấn luyện trên Foody $\to$ Kiểm thử trên Foody.
3. **$A \to B$ (Cross-domain):** Huấn luyện trên Sách $\to$ Kiểm thử trên Foody.
4. **$B \to A$ (Cross-domain):** Huấn luyện trên Foody $\to$ Kiểm thử trên Sách.

### 6.2. Chỉ số Đánh giá Cốt lõi
- **Chỉ số chính:** **Macro-F1 Score** (trung bình cộng không trọng số F1 của 3 lớp).
- **Mức độ suy giảm hiệu năng:**
  $$\text{Performance Drop} = \text{Macro-F1 (In-domain)} - \text{Macro-F1 (Cross-domain)}$$

### 6.3. Bảng Kết quả Thực nghiệm Thực tế (PhoBERT + Focal Loss)
| Kịch bản | Miền | Loại | Macro-F1 | Accuracy | F1 (Negative) | F1 (Neutral) | F1 (Positive) | Performance Drop ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$A \to A$** | Sách $\to$ Sách | In-domain | **0.8109** | 89.16% | 0.9122 | 0.5850 | 0.9355 | — |
| **$B \to B$** | Nhà hàng $\to$ Nhà hàng | In-domain | **0.8588** | 92.72% | 0.9431 | 0.6734 | 0.9601 | — |
| **$A \to B$** | Sách $\to$ Nhà hàng | Cross-domain | **0.8211** | 91.23% | 0.9126 | 0.5969 | 0.9538 | **-0.0102** (Tăng nhẹ) |
| **$B \to A$** | Nhà hàng $\to$ Sách | Cross-domain | **0.7596** | 85.02% | 0.8714 | 0.4984 | 0.9091 | **+0.0992** (Tụt ~9.9%) |

### 6.4. Hiện tượng Bất đối xứng Chuyển miền (Domain Asymmetry)
- Chiều $A \to B$ không bị suy giảm hiệu năng do miền Sách rèn luyện cho mô hình khả năng nhận diện cảm xúc tổng quát, mạch lạc.
- Chiều $B \to A$ bị suy giảm ~9.9% do:
  1. *Độ lệch độ dài văn bản (Length Shift):* Nhà hàng dài trung bình 65 từ; Sách ngắn 33 từ $\to$ thiếu hụt ngữ cảnh cho cơ chế Self-Attention.
  2. *Độ lệch từ vựng chuyên ngành (Vocabulary Shift):* Sách chứa các khái niệm xuất bản vật lý (*màng co, cấn gáy, nxb, bookmark, shipper*) hoàn toàn vắng bóng ở miền Ẩm thực.

---

## 7. HỆ THỐNG PHÂN LOẠI LỖI (ERROR TAXONOMY)

Các ca dự đoán sai trong thực tế được tự động trích xuất và phân loại vào 8 nhóm lỗi định danh tại [`src/error_analysis.py`](../src/error_analysis.py):
1. **Domain-specific Vocabulary:** Chứa thuật ngữ đặc thù của miền chưa từng gặp ở tập huấn luyện.
2. **Sarcasm / Nghịch ngữ:** Lối nói mỉa mai hoặc khen ngợi kiểu Gen Z (*"bực mình quá đi, gì mà nhân viên dễ thương..."*).
3. **Phủ định phức tạp (Negation):** Cấu trúc đảo ngược nghĩa hoặc phủ định tình huống tìm đường/thời gian.
4. **Mixed Sentiment:** Vừa khen vừa chê liên kết bởi từ đối lập (*"nhưng, tuy nhiên"*).
5. **Aspect Conflict (Shipper vs Sản phẩm):** Khen nội dung sản phẩm nhưng chê thái độ vận chuyển.
6. **Rating và Nội dung mâu thuẫn:** Nhãn gốc bị sai lệch do người dùng bấm nhầm số sao.
7. **Review quá ngắn & Teencode:** Các câu 2–3 từ thiếu thông tin ngữ cảnh.
8. **Domain Shift tổng quát:** Khác biệt phong cách diễn đạt cảm xúc giữa hai miền.
