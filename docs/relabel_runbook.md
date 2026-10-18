# Quy trình gán lại nhãn "vùng xám" theo nội dung

## Vì sao cần

Kiểm tra nhãn thủ công D3 (`data/processed/label_audit_{A,B}.csv`, 100 mẫu mỗi lớp) cho thấy:

| | Negative | Neutral | Positive |
|---|---|---|---|
| A (sách, nhãn suy từ số sao) | 96% đúng | **29%** | 82% |
| B (nhà hàng, nhãn có sẵn) | 96% | **58%** | 100% |

- Ở A, review 3★ phần lớn là **chê**.
- Ở B, Neutral phần lớn là **khen có chê nhẹ**.
- Nghĩa của "Neutral" ở hai miền ngược nhau, nên kết quả cross-domain bị lẫn với lỗi nhãn.

**Quyết định của nhóm:** đọc lại toàn bộ "vùng xám" và gán nhãn theo nội dung. Review nào thực chất là chê hoặc khen thì **chuyển sang lớp đúng**, không bỏ đi.

| Vùng xám | Số review | Lý do |
|---|---|---|
| A: review 3★ | ~3.800 | Neutral chỉ đúng 29% |
| A: review 4★ | ~7.700 | Nơi chứa phần lớn Neutral thật (khen chê lẫn lộn) |
| B: nhãn Neutral | 6.498 | Neutral chỉ đúng 58% |
| **Tổng** | **18.027** | 91 batch × 200 review |

A 1–2★, A 5★, B Negative và B Positive giữ nguyên nhãn, vì D3 cho thấy các nhóm này đáng tin.

## Các bước

### 1. Xuất batch (đã chạy sẵn, chỉ chạy lại khi dữ liệu gốc thay đổi)
```bash
python -X utf8 -m src.relabel export
```
Lệnh này tạo `data/relabel/batches/{A,B}_NNN.jsonl`, mỗi dòng có dạng `{"id", "text"}`. **Nhãn gốc không được xuất ra**, để AI gán nhãn không bị kéo theo nhãn cũ.

### 2. Gán nhãn bằng AI (chọn một trong hai cách)

Prompt dùng chung cho cả hai cách: [`docs/relabel_prompt.md`](relabel_prompt.md). Prompt gồm định nghĩa nhãn, ví dụ và định dạng đầu ra. **Không sửa prompt giữa chừng.** Nếu cần sửa thì chạy lại từ đầu, để mọi batch được gán theo cùng một tiêu chí.

Kết quả của mỗi batch ghi vào `data/relabel/results/<cùng tên batch>.jsonl`. Mỗi dòng có dạng:
```json
{"id": "tiki_10002113", "label": 1, "note": "Mới đọc, thấy khá hữu ích, chưa đánh giá rõ"}
```

**Cách A: script tự động (khuyên dùng)**, chạy được với mọi API tương thích OpenAI (DeepSeek, Gemini, OpenAI, OpenRouter, Ollama…):
```bash
pip install openai
export RELABEL_API_KEY=...   RELABEL_BASE_URL=...   RELABEL_MODEL=...
python -X utf8 scripts/run_relabel_api.py --only A_000 B_000   # chạy thử
python -X utf8 scripts/run_relabel_api.py                      # chạy phần còn lại; dừng rồi chạy lại vẫn tiếp tục được
```
- Script gửi 50 review mỗi lần gọi, với `temperature=0`.
- Lần gọi nào thiếu id hoặc nhãn không hợp lệ sẽ tự thử lại.
- Batch chỉ được ghi file khi đã đủ kết quả.

**Cách B: thủ công hoặc qua một agent khác.** Với mỗi file batch, đưa nội dung `relabel_prompt.md` làm system prompt và nội dung file batch làm tin nhắn. Lưu nguyên câu trả lời vào `data/relabel/results/<tên batch>.jsonl`. Nếu model hay bỏ sót dòng thì gửi mỗi lần 50 dòng.

**Chọn model:** cần đọc tiếng Việt tốt; nên dùng `temperature=0`. Tổng cộng khoảng 18.000 review, cỡ 2–3 triệu token đầu vào.

### 3. Chạy thử và kiểm tra chất lượng (BẮT BUỘC trước khi chạy hết)
Chạy khoảng 4 batch (ví dụ `A_000 A_001 B_000 B_001`), rồi:
```bash
python -X utf8 -m src.relabel validate --partial
python -X utf8 -m src.relabel qa --partial
```
`qa` in ra ba thứ:
- Bảng chuyển nhãn (nhãn gốc → nhãn mới) theo từng miền.
- **Độ đồng thuận với D3** trên các review vừa nằm trong vùng xám vừa có trong file chấm D3. Đồng thuận nghĩa là: D3 nói đúng thì nhãn mới giữ nguyên, D3 nói sai thì nhãn mới phải khác nhãn gốc. Mục tiêu ≥ 80%.
- File `data/relabel/review_sample.csv` gồm 50 mẫu mỗi miền để người duyệt.

Mở `review_sample.csv`, đọc cột `text`, điền `human_label` (0/1/2), rồi tính tỷ lệ `human_label == new_label`. **Nếu dưới 80% thì đổi model hoặc chỉnh prompt rồi chạy lại từ đầu**, trước khi chạy hết 91 batch.

### 4. Chạy hết và gộp kết quả
```bash
python -X utf8 -m src.relabel validate      # mọi batch đều phải có kết quả hợp lệ
python -X utf8 -m src.relabel merge         # ghi data/relabel/relabel_labels.csv
python -X utf8 -m src.relabel qa            # đồng thuận D3 trên toàn bộ + xuất mẫu duyệt cuối
```

### 5. Build lại dữ liệu
```bash
python -X utf8 -m src.build_dataset
```
- Nếu có file `data/relabel/relabel_labels.csv`, pipeline tự áp nhãn mới theo `id` **trước** bước lọc, bỏ trùng và chia tập.
- Cột `label_source` trong dữ liệu đầu ra cho biết nguồn nhãn của từng dòng: `rating` (từ số sao), `dataset` (nhãn gốc của res1) hoặc `content` (gán lại theo nội dung).
- Các kiểm tra tự động vẫn áp dụng: mỗi lớp trong test ≥ 250 mẫu, train mỗi lớp ≥ 1.000, không rò rỉ, train A = train B.
- Nếu một kiểm tra không đạt (ví dụ Neutral của A quá ít) thì **dừng lại và báo cho nhóm quyết định**. Không tự hạ ngưỡng.
- `label_audit_*.csv` không bị ghi đè. Sau khi build lại nên xóa (hoặc đổi tên) hai file này rồi build lại một lần nữa, để có mẫu D3 mới cho vòng kiểm tra lại.

### 6. Ghi vào báo cáo
- Mô tả cách gán lại nhãn: phạm vi vùng xám, prompt và model đã dùng.
- Độ đồng thuận với D3 và với người duyệt (số mẫu và tỷ lệ).
- Bảng chuyển nhãn (gốc → mới) của từng miền.
- Kết quả D3 trước và sau khi gán lại.
- Kết quả D3 trước khi gán lại là một phát hiện riêng: nhãn suy từ số sao kém tin cậy ở lớp giữa, và nghĩa của Neutral khác nhau giữa hai miền.

## File liên quan

| File | Vai trò |
|---|---|
| `src/relabel.py` | Các lệnh `export`, `validate`, `merge`, `qa` |
| `scripts/run_relabel_api.py` | Script chạy tự động qua API |
| `docs/relabel_prompt.md` | Prompt và tiêu chí gán nhãn |
| `data/relabel/batches/` | Đầu vào cho AI (commit vào repo) |
| `data/relabel/results/` | Đầu ra của AI (commit để tái lập) |
| `data/relabel/relabel_labels.csv` | Nhãn mới đã gộp; `build_dataset` tự đọc file này |
