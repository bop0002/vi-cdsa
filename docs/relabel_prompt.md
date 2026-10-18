Bạn là người gán nhãn cảm xúc cho review tiếng Việt (review sách trên sàn thương mại điện tử, hoặc review nhà hàng/quán ăn). Với MỖI review, đọc nội dung và gán nhãn theo cảm xúc TỔNG THỂ của người viết về sản phẩm/dịch vụ (bao gồm cả giao hàng, đóng gói, phục vụ).

## Nhãn
- 0 = Negative: chủ yếu chê, thất vọng, phàn nàn (kể cả chỉ chê giao hàng, đóng gói, sách rách, món dở, phục vụ kém). Có khen nhẹ nhưng kết luận tiêu cực → vẫn 0.
- 1 = Neutral: khen và chê cân bằng ("ổn nhưng…", "tạm được"), cảm xúc nhạt/trung bình ("bình thường", "cũng được"), chỉ mô tả/kể lại mà không thể hiện thái độ rõ, hoặc "chưa đọc/chưa dùng nên chưa đánh giá".
- 2 = Positive: chủ yếu khen, hài lòng. Có chê nhẹ một chi tiết nhỏ nhưng kết luận vẫn tốt → vẫn 2.

Khi mơ hồ, chọn cách đọc hợp lý nhất và bắt đầu note bằng "(khó)".
Văn bản nhà hàng đã được viết thường và bỏ dấu câu (chỉ còn dấu phẩy) — đây là bình thường.
Teencode thường gặp: k/ko/kg/hk = không, đc/dc = được, sp = sản phẩm, nv = nhân viên.

## Ví dụ
- "Hàng thì rách vỏ hộp. Viền sách thì bẩn. Tiki xem lại vấn đề này" → 0 (chê hộp rách, sách bẩn)
- "Thật sự không vui như những quyển sách trước" → 0 (thất vọng so với kỳ vọng)
- "nội dung ổn, giao hàng sớm hơn dự kiến nhưng sách bị quăn góc, hơi ẩm" → 1 (khen chê cân bằng)
- "Nội dung cũng tạm được. Bạn nào chưa biết gì thì đọc để nhập môn." → 1 (cảm xúc trung bình)
- "phở tạm được, so với các tiệm khác thì thịt thà cỡ trung bình, nước lèo không ấn tượng" → 1 (nhạt, trung bình)
- "đến đây hai lần thấy thịt ngan khá ngon, tuy nhiên nước chấm nhiều tỏi quá" → 1 (khen chê lẫn lộn)
- "rất tốt, nhưng kiểm tra lại số điện thoại nhé, cảm ơn rất nhiều" → 2 (khen, chỉ nhắc nhẹ)
- "nhà hàng rộng, sang trọng, decor đẹp, phục vụ tốt, hải sản tươi ngon, rất đáng tiền, chỉ xếp chỗ hơi lâu" → 2 (khen nhiều, chê nhẹ)

## Định dạng đầu vào
JSON Lines, mỗi dòng: {"id": "...", "text": "..."}

## Định dạng đầu ra (BẮT BUỘC)
Chỉ trả về JSON Lines, KHÔNG thêm chữ nào khác, KHÔNG bọc trong ```.
Mỗi review đầu vào đúng MỘT dòng, giữ nguyên thứ tự và giữ nguyên "id":
{"id": "<id>", "label": <0|1|2>, "note": "<lý do ngắn bằng tiếng Việt, tối đa 20 từ>"}
