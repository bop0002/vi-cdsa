# Quy tắc gán nhãn cảm xúc

Áp dụng cho toàn bộ dữ liệu của dự án:

- **Domain A**: review sách Tiki.
- **Domain B**: review nhà hàng (res1 và Foody).

Có 3 nhãn: **0 = Negative**, **1 = Neutral**, **2 = Positive**.

Nguồn của tài liệu này:

- Tiêu chí gốc lấy từ prompt gán nhãn `docs/relabel_prompt.md`, dùng y nguyên cho cả 264 batch của hai đợt.
- Các quy tắc chi tiết ở mục 4 được rút ra từ 52.537 nhãn đã gán, lưu trong `data/relabel/relabel_labels.csv`.

---

## 1. Dòng nào lấy nhãn từ số sao, dòng nào phải đọc nội dung

Cột `label_source` trong file parquet cho biết nhãn của mỗi dòng đến từ đâu: `rating` (số sao hoặc điểm), `dataset` (nhãn có sẵn của res1) hoặc `content` (gán theo nội dung).

| Nguồn | Vùng tin cậy: giữ nhãn gốc | Vùng xám: **bắt buộc gán theo nội dung** |
|---|---|---|
| Tiki sách (A) | 1–2★ → 0; 5★ → 2 | **3★ và 4★** |
| res1 (B) | Negative, Positive có sẵn | **Neutral có sẵn** |
| Foody (B), thang 10 | < 3.0 → 0; > 7.5 → 2 | **3.0 – 7.5** (Foody trộn thang 10 và thang 5★ nên khoảng này không suy ra được) |

Vì sao phải đọc lại vùng xám: trong lần kiểm tra thủ công D3 (100 mẫu mỗi lớp), nhãn Neutral suy từ số sao chỉ đúng **29%** ở A và **58%** ở B.

---

## 2. Định nghĩa nhãn

Gán theo **cảm xúc tổng thể của người viết** về sản phẩm hoặc dịch vụ. Điều này **tính cả** giao hàng, đóng gói và phục vụ, không chỉ nội dung sách hay món ăn.

| Nhãn | Khi nào gán |
|---|---|
| **0 = Negative** | Chủ yếu chê, thất vọng hoặc phàn nàn. **Vẫn là 0** khi chỉ chê một khía cạnh ngoài lề, ví dụ giao hàng, đóng gói, sách rách, món dở, phục vụ kém. Có khen nhẹ nhưng **kết luận tiêu cực** thì vẫn là 0. |
| **1 = Neutral** | Có một trong các dạng: (a) khen và chê **cân bằng** ("ổn nhưng…"); (b) cảm xúc **nhạt hoặc trung bình** ("bình thường", "tạm được", "cũng được"); (c) **chỉ mô tả hoặc kể lại**, không thể hiện thái độ; (d) **chưa đọc hoặc chưa dùng** nên chưa đánh giá. |
| **2 = Positive** | Chủ yếu khen, hài lòng. Có chê nhẹ một chi tiết nhỏ nhưng **kết luận vẫn tốt** thì vẫn là 2. |

---

## 3. Quy trình quyết định

```
Review có phải quảng cáo, spam hoặc chuỗi vô nghĩa không?
 ├─ Có  → 1, note bắt đầu bằng "(spam)". Dòng này sẽ bị loại khỏi dataset.
 └─ Không
     Người viết có bày tỏ thái độ không?
      ├─ Không (chỉ mô tả, hỏi thông tin, chưa đọc/chưa dùng) → 1
      └─ Có
          Kết luận hoặc cảm xúc chiếm ưu thế là gì?
           ├─ Chê hoặc thất vọng là chính           → 0
           ├─ Khen là chính, chỉ góp ý nhỏ          → 2
           └─ Ngang nhau, hoặc chỉ "tạm", "bình thường" → 1
Còn phân vân → chọn cách hiểu hợp lý nhất, note bắt đầu bằng "(khó)".
```

---

## 4. Quy tắc chi tiết cho các trường hợp hay gặp

1. **Giao hàng và đóng gói được tính vào cảm xúc.** Ví dụ: "Sách đẹp nhưng bị móp góc, giao chậm" có thể là 0 hoặc 1, tuỳ phần nào nặng hơn. Đây là nhóm lớn nhất, có 9.183 nhãn nhắc tới giao hàng hoặc đóng gói.
2. **Khi review vừa khen vừa chê, câu kết luận quyết định nhãn.** "Giao nhanh, sách cứng cáp nhưng sách bị lỗi" → 0, vì lỗi sản phẩm là điều người viết muốn nói. "Bìa dính chút sơn nhưng không sao, sẽ ủng hộ tiếp" → 2.
3. **Chưa đọc hoặc chưa dùng:** mặc định gán **1** (80% các trường hợp đã gán như vậy). Ngoại lệ:
   - Có chê rõ về hình thức hoặc giao hàng thì gán 0. Ví dụ: "chưa đọc nhưng đóng gói tệ".
   - Có khen rõ thì gán 2. Ví dụ: "chưa đọc nhưng bìa đẹp, rất hài lòng".
4. **Không như kỳ vọng, thua lần trước** → thường gán **0**. Ví dụ: "không hay như mình nghĩ". Nếu có kèm "tạm được" thì gán 1.
5. **Sai mô tả sản phẩm** (sai kích thước, bìa cứng hay mềm, sai phiên bản) → **0**.
6. **"Tạm được", "bình thường", "cũng ổn"** đứng một mình → **1**. Đi kèm lời chê thì gán 0.
7. **Giá cả** chỉ là một khía cạnh như các khía cạnh khác:
   - "Giá cao mà đồ ăn bình thường" → 0.
   - "Giá rẻ, đầy đặn nhưng không ngon lắm" → 1.
   - "Giá tốt, nên mua" → 2.
8. **Hỏi thông tin, chỉ ghi địa chỉ, kể lịch trình** mà không có thái độ → **1**.
9. **Lời khuyên hoặc góp ý nhẹ** trong một review đang khen ("nên bọc màng kính", "kiểm tra lại số điện thoại nhé") không làm đổi nhãn khen → **2**.
10. **Không dùng số sao để suy nhãn.** Batch gán nhãn cố ý không có số sao, chỉ dựa vào nội dung.
11. **Định dạng văn bản không ảnh hưởng đến nhãn.**
    - res1 đã được viết thường và bỏ dấu câu.
    - Foody còn nguyên chữ hoa và dấu câu.
    - Teencode: `k/ko/kg/hk` = không, `đc/dc` = được, `sp` = sản phẩm, `nv` = nhân viên.

---

## 5. Ví dụ chuẩn

| Review | Nhãn | Lý do |
|---|:-:|---|
| Hàng thì rách vỏ hộp. Viền sách thì bẩn. Tiki xem lại vấn đề này | 0 | Chê hộp rách, sách bẩn |
| Thật sự không vui như những quyển sách trước | 0 | Thất vọng so với kỳ vọng |
| Sách mô tả kích thước 14.5x20.5 nhưng khi nhận thì không đúng | 0 | Sai mô tả |
| giá mắc, ăn không ngon nói chung ko quay trở lại | 0 | Chê, kết luận không quay lại |
| nội dung ổn, giao hàng sớm hơn dự kiến nhưng sách bị quăn góc, hơi ẩm | 1 | Khen chê cân bằng |
| Nội dung cũng tạm được. Bạn nào chưa biết gì thì đọc để nhập môn. | 1 | Cảm xúc trung bình |
| phở tạm được, thịt cỡ trung bình, nước lèo không ấn tượng | 1 | Nhạt, trung bình |
| Chưa đọc nên chưa review được | 1 | Chưa đánh giá |
| Mng cho mình hỏi địa điểm này còn hoạt động không ạ? | 1 | Chỉ hỏi thông tin |
| rất tốt, nhưng kiểm tra lại số điện thoại nhé, cảm ơn rất nhiều | 2 | Khen, chỉ nhắc nhẹ |
| nhà hàng rộng, decor đẹp, hải sản tươi ngon, rất đáng tiền, chỉ xếp chỗ hơi lâu | 2 | Khen nhiều, chê nhẹ |
| sách đẹp, giấy tốt, trừ điểm vì shop k bọc giấy kính cho bìa | 2 | Khen, góp ý nhỏ |

---

## 6. Định dạng kết quả

Đầu vào là JSON Lines, mỗi dòng `{"id": "...", "text": "..."}`. Đầu ra cũng là JSON Lines, **mỗi id đúng một dòng**, giữ nguyên thứ tự:

```json
{"id": "tiki_10002113", "label": 2, "note": "Khen sách khá hữu ích"}
```

- `label`: phải là số `0`, `1` hoặc `2`, không phải chuỗi.
- `note`: lý do ngắn bằng tiếng Việt, **tối đa 20 từ, không được để trống**.
- Review mơ hồ: `note` bắt đầu bằng `(khó)`.
- Spam: gán `label = 1`, `note` bắt đầu bằng `(spam)`. Pipeline sẽ loại các dòng này (`src/relabel.py: is_spam`).
- Kiểm tra định dạng: `python -X utf8 -m src.relabel validate`.

---

## 7. Kiểm soát chất lượng

- **Không sửa tiêu chí giữa chừng.** Nếu đổi prompt hoặc quy tắc thì phải gán lại toàn bộ để mọi batch nhất quán.
- Gán nhãn với `temperature = 0`, mỗi lần gửi 50 review để tránh model bỏ sót hoặc lệch dòng.
- **Ngưỡng chấp nhận:** đồng thuận với lần kiểm tra thủ công D3 phải **≥ 80%**. Đợt 1 đạt **A 92%, B 82%**. Ngoài ra cần chấm tay mẫu `data/relabel/review_sample.csv` (50 mẫu mỗi domain), cũng với ngưỡng ≥ 80%.

### Kết quả thực tế (52.537 dòng vùng xám)

| Domain | Nhãn gốc | → 0 | → 1 | → 2 |
|---|---|--:|--:|--:|
| A | 3★ (Neutral) | 5.510 | 2.347 | 618 |
| A | 4★ (Positive) | 4.008 | 5.498 | 7.497 |
| B | Negative (Foody vùng xám) | 2.474 | 93 | 17 |
| B | Neutral | 5.638 | 4.235 | 6.093 |
| B | Positive (Foody vùng xám) | 1.038 | 2.381 | 5.090 |

- 3.827 dòng (7,3%) được đánh dấu `(khó)`, và 42 dòng bị đánh dấu `(spam)`.
- Kết luận chính: **review 3★ của sách phần lớn là chê**, còn **Neutral của nhà hàng phần lớn là khen có chê nhẹ**. Nếu dùng số sao thì nghĩa của "Neutral" sẽ ngược nhau giữa hai domain.

---

## 8. Hạn chế

- Nhãn vùng xám do LLM gán. Nhãn của cả lớp Neutral trong dataset đều từ nguồn này (`label_source = content`).
- Các dòng `(khó)` là nơi nhãn kém chắc chắn nhất, nên ưu tiên kiểm tra khi phân tích lỗi mô hình.
- Ranh giới giữa 0 và 1 khi review **chỉ chê giao hàng hoặc đóng gói** là trường hợp không nhất quán nhất. Nhóm này có 3.733 dòng gán 0 và 2.603 dòng gán 1.
