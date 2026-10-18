# Định nghĩa tập dữ liệu sạch (Clean Dataset Spec)

Áp dụng cho mọi nguồn dữ liệu (Domain A, Domain B). Một nguồn chỉ được đưa vào thí nghiệm khi **đạt toàn bộ tiêu chí bắt buộc (B)** sau khi làm sạch.

---

## 1. Schema chuẩn

Mỗi dòng = 1 review, lưu `data/processed/<domain>_<split>.parquet`:

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | str | `<source>_<review_id gốc>`, duy nhất toàn bộ dữ liệu |
| `domain` | str | `A` (e-commerce) / `B` (hotel-restaurant) |
| `source` | str | `tiki_book`, `res1` |
| `group_id` | str | Sản phẩm / khách sạn / nhà hàng mà review thuộc về (dùng để chia tập). Với res1: `group_id = id` (không có nhóm thật) |
| `category` | str | Ngành hàng / loại hình: không có trong cặp Book–res1 (không xuất) |
| `rating` | int | Rating gốc (giữ để truy vết); `rating = -1` với res1 (không có rating) |
| `label` | int | 0 = Negative, 1 = Neutral, 2 = Positive |
| `split` | str | `train` / `val` / `test` |
| `text_raw` | str | Nội dung gốc, chưa sửa |
| `text_clean` | str | Text sạch dùng chung (mục 3) |
| `text_ml` | str | Đầu vào cho TF-IDF (mục 4) |
| `text_dl` | str | Đầu vào cho PhoBERT (mục 4) |
| `n_words` | int | Số từ của `text_clean` |

---

## 2. Tiêu chí

### 2.1. Cấp dòng: dòng nào được giữ (B = bắt buộc)

| # | Tiêu chí | Lý do |
|---|---|---|
| R1 (B) | **Không chứa text do nền tảng tự sinh** từ rating (vd. Tiki: "Cực kì hài lòng.", "Bình thường."). Phải cắt trước mọi bước khác | Lộ nhãn: model học tiêu đề thay vì nội dung |
| R2 (B) | Sau khi cắt R1, text không rỗng và có **≥ 3 từ** | Review "ok", "tốt" không mang đủ thông tin, và trùng lặp rất nhiều |
| R3 (B) | Rating hợp lệ (1–5 hoặc thang của nguồn) và ánh xạ được về 3 lớp | — |
| R4 (B) | **Chủ yếu tiếng Việt**: có ít nhất 1 ký tự có dấu tiếng Việt, hoặc qua bộ phát hiện ngôn ngữ | Đúng phạm vi đề tài |
| R5 (B) | **Không trùng nội dung** trong cùng domain (so sánh trên text đã chuẩn hóa: NFKC, lowercase, gộp khoảng trắng, bỏ dấu câu cuối). Mỗi text chỉ giữ 1 bản | Tránh trùng giữa train và test, tránh thổi phồng tần suất |
| R6 (B) | **Không xung đột nhãn**: text giống nhau nhưng có rating khác lớp thì bỏ cả nhóm | Nhãn mâu thuẫn |
| R7 | Bỏ spam: cùng user, cùng text, dán ở nhiều sản phẩm | Nhiễu |
| R8 | Không phải nội dung rác: chỉ có emoji, chỉ có ký tự lặp, chỉ có URL hoặc số | — |

### 2.2. Cấp tập dữ liệu

| # | Tiêu chí | Ngưỡng đề xuất |
|---|---|---|
| D1 (B) | Có nguồn và cách thu thập rõ ràng, dùng cho mục đích học tập | Ghi trong README |
| D2 (B) | **Đủ dữ liệu ở lớp nhỏ nhất**: test set mỗi lớp ≥ 300 mẫu, train mỗi lớp ≥ 1.000 mẫu | Để Macro-F1 ổn định và PhoBERT học được lớp nhỏ |
| D3 (B) | **Nhãn đáng tin**: đọc thủ công 100 mẫu ngẫu nhiên mỗi lớp, ≥ 85% đúng nhãn (riêng Neutral ≥ 75%) | Rating ≠ sentiment của text |
| D4 | **Đa dạng**: không sản phẩm/địa điểm nào chiếm > 2% domain (giới hạn số review mỗi `group_id`) | Tránh model học thuộc 1 sản phẩm |
| D5 (B) | **Không rò rỉ khi chia tập**: không trùng text giữa train/val/test; không trùng text giữa Domain A và B | Nguyên tắc 7 trong kế hoạch |
| D6 | Phân bố lớp, độ dài, số nhóm được ghi lại trước và sau mỗi bước | Nguyên tắc 9 |
| D7 (B) | Tái hiện được: cố định seed (42), chạy lại script ra đúng kết quả | Nguyên tắc 10 |

---

## 3. `text_clean`: tiền xử lý chung (ML và DL dùng chung)

Thực hiện bởi `src/text_norm.clean_text`, theo đúng thứ tự:

1. Unicode NFKC.
2. Cắt tiêu đề Tiki tự sinh (chỉ khi theo sau là `.` `!` `:` `-` hoặc hết text) và cụm "(*) Đánh giá không tính điểm" (R1).
3. Chuyển chữ thường.
4. Xóa URL.
5. Xóa emoji.
6. `. ! ? ; : …` và xuống dòng → dấu phẩy; `_` → khoảng trắng.
7. Xóa mọi ký tự khác ngoài chữ, số, khoảng trắng, dấu phẩy và `/`.
8. Chữ cái lặp ≥ 3 lần → 2 lần (`quáaaaa` → `quáa`).
9. Gộp dấu phẩy và khoảng trắng liên tiếp.
10. Teencode theo từ điển `TEENCODE` (gồm `kg` → `không` trừ khi đứng sau số; `k` → `không` trừ khi dạng giá tiền: sau số và ngay trước dấu phẩy hoặc hết câu, ví dụ `27 k,`).

## 4. Biểu diễn riêng theo loại mô hình (cặp Book–res1)

res1 đã được tiền xử lý sẵn (chữ thường, chỉ còn dấu phẩy, không emoji), nên **cả hai miền** được đưa về đúng định dạng này để khác biệt định dạng không bị lẫn vào domain shift.

| | `text_ml` (TF-IDF + LR/SVC) | `text_dl` (PhoBERT) |
|---|---|---|
| Hoa/thường | lowercase | lowercase (do B không có chữ hoa) |
| Unicode | NFKC (gộp ký tự toán học/full-width) | NFKC |
| Teencode | từ điển chung `src/text_norm.TEENCODE` | như ML |
| Emoji | bỏ | bỏ |
| Dấu câu | bỏ dấu phẩy (giữ `/`, bộ tách token của TF-IDF tự tách) | chỉ giữ dấu phẩy (`. ! ? ; :` và xuống dòng → `,`) |
| Tách từ | VnCoreNLP (RDRSegmenter) | VnCoreNLP (RDRSegmenter) |
| Tên nền tảng | giữ nguyên | giữ nguyên |
| Độ dài | không giới hạn | cắt tối đa 256 subword |

Dùng chung một từ điển teencode và một hàm emoji cho cả hai nhánh, để khi so sánh ML với DL thì khác biệt đến từ mô hình chứ không phải từ cách tiền xử lý.

---

## 5. Kiểm tra sau xử lý (checklist tự động)

- [ ] Không còn text nền tảng tự sinh ở đầu review (regex = 0 kết quả)
- [ ] `n_words >= 3` với mọi dòng
- [ ] `text_clean` không trùng trong domain; không trùng giữa train/val/test; không trùng giữa A và B
- [x] Không có `group_id` nào vượt ngưỡng D4 (đã cài trong `src/checks.py`)
- [ ] Mỗi lớp trong test ≥ 300 mẫu
- [ ] In phân bố lớp và độ dài của từng split → `data/processed/stats.json`

## 6. Quyết định cho cặp Book–res1 (2026-10-16)

- Domain A = `book_comment_tiki.csv`; Domain B = `res1`. Nhãn res1 được map `{0:0, 1:2, 2:1}`.
- res1 giữ split gốc; mẫu trùng giữa các split giữ bản trong train.
- Book chia 80/10/10 theo `product_id` (StratifiedGroupKFold, seed 42).
- Chỉ cân bằng train: train A được downsample (giữ tỷ lệ lớp) về đúng số mẫu train B; val/test A giữ nguyên để đạt D2.
- Mỗi miền có thêm `test_balanced` (mỗi lớp bằng lớp nhỏ nhất) để tách ảnh hưởng của label shift.
- Teencode: `kg` → `không`, trừ khi đứng sau số (`2 kg`, `6kg` là cân nặng). Ở book 36/40 lần `kg` mang nghĩa "không".
- Teencode: `k` sau số và ngay trước dấu phẩy/hết câu là giá tiền nên giữ nguyên (`coca 27 k,`); các trường hợp khác (`quyển 1 k hiểu`) đổi thành `không`. Ảnh hưởng 11 dòng. Review bắt đầu bằng "Cực kỳ hài lòng" + xuống dòng (2 dòng ở A) được giữ nguyên, nhất quán với quyết định giữ review người dùng tự viết.
- Chuẩn hóa NFKC chạy trước bước cắt tiêu đề Tiki (sửa lỗi review ở dạng NFD giữ lại tiêu đề).
- Review do người dùng tự viết bắt đầu bằng "hài lòng, …", "bình thường, …" được giữ; checklist chỉ báo lỗi khi cả review chỉ là cụm tiêu đề.
- VnCoreNLP không ghép "giao hàng" thành `giao_hàng`; model được tải bằng urllib (Windows không có wget).
- Kết quả build (seed 42): train A = train B = 25.529; A val 7.203 / test 6.639; B val 3.153 / test 3.122; test_balanced A 1.098, B 1.797. Bước bỏ trùng giữa A và B bỏ 6 dòng mỗi bên.
- Hạn chế: A chỉ có sách, B chỉ có nhà hàng; res1 không có `group_id` nên không chia theo group được; nguồn và license của hai bộ cần ghi rõ.
