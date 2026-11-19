# LifeMate — bản giới thiệu kỹ thuật

Dự án chuyển yêu cầu giọng nói tiếng Việt thành hành động trên ứng dụng. Phần công việc được trình bày ở đây là tích hợp thí nghiệm, đánh giá ASR, chẩn đoán lỗi, kiểm nguồn gốc checkpoint và ghi nhật ký thực thi công cụ. W50 là adapter do nhóm cung cấp; Qwen/PhoWhisper và các model nền được ghi công riêng.

## Các điểm nên xem

1. [Nhật ký xử lý lỗi](engineering_journal.md): triệu chứng → bằng chứng → cách sửa → kiểm tra → giới hạn. Có các ca NumPy ABI, mask không contiguous, OOM, AMP/resume, sai model/cache và telemetry EOS gây hiểu nhầm.
2. [Kết quả](results.md): WER đi cùng latency; điểm model khác điểm hiệu chỉnh thủ công. Không lấy số đẹp nhất làm kết quả của cả pipeline.
3. [Mã nguồn và kiểm thử](../src/lifemate_portfolio):16 CPU tests cho cache/checkpoint contract và nhật ký công cụ. Chạy được bằng Python, không cần model hoặc dữ liệu cuộc thi.
4. [Kiến trúc](architecture.md), [tái lập](reproducibility.md) và [nguồn gốc](provenance.md).

**Input của bản minh họa:** dữ liệu giả tự tạo, kế hoạch công cụ viết sẵn; không nhận dạng âm thanh và không dùng LLM. **Input của thí nghiệm thật:** audio current/history, context, schema và SDK của BTC, model base/adapter đúng phiên bản. Hai GPU mặc định **2 × T4**; máy chấm mục tiêu **1 × A30**.

Kết quả đã lưu: large beam 5 đạt9,96%WER trên154 yêu cầu hiện tại; medium11,40% trên cùng benchmark. Large có p95ASR cả hội thoại33,18giây trênT4 nên chưa thể chọn chỉ dựa vào WER. W50+Zipformer63,50 và W50+mediumV2 53,87 là điểm người dùng báo; chúng không dùng cùng checkpoint medium trong bảng WER.81,99 là nhánh chỉnh đầu ra theo mẫu, **không phải điểm Qwen tự chạy**.

Repo này không chứa dữ liệu cuộc thi, weights, prediction rows hoặc script sửa theo public ID. Notebook tham khảo đã xóa output và mặc định chỉ preflight. Chưa tuyên bố đạt30giây/mẫu trênA30, chưa có score private hoặc kết quả mới của các thử nghiệm prompt/thinking.
