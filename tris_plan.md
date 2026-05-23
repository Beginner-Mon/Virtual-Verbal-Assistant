# Plan: VVA (KineticChat) Re-Architecture — LangGraph Multi-Agent Supervisor (v2.3)

> Architect: T | Developer: N | Date: 2026-05-22
> Status: **v2.3 — Concurrent MCP Execution, Upstream Approval Gate, Grader Max Loop=1, No Validator**

---

## Context & Key Decisions

Bản kế hoạch v2.3 này chính thức thống nhất luồng thực thi Agent dựa trên mô hình **Plan-and-Execute** chặt chẽ do Architect T đề xuất, tối ưu hóa sâu hơn để loại bỏ các node trung gian và xử lý triệt để các phản biện kỹ thuật về hạ tầng và trải nghiệm người dùng (UX).

### Các thay đổi cốt lõi trong v2.3:
1. **Concurrent MCP Execution:** Các công cụ (NVIDIA Kimodo, Web Search...) được kích hoạt song song qua `asyncio.gather`. Tốc độ xử lý của Node phụ thuộc vào tool chậm nhất. Khoảng thời gian chờ được lấp đầy bằng các trạng thái cập nhật liên tục gửi về client qua Server-Sent Events (SSE).
2. **Upstream Approval Gate:** Di chuyển cổng phê duyệt lên trước bước gọi tool nặng. Hệ thống hỏi quyền truy cập một lần cho các tác vụ cần thiết. User đồng ý mới kích hoạt hạ tầng tính toán ngầm.
3. **Grader Max Loop = 1:** Khóa cứng số vòng lặp tối đa của Grader node để bảo vệ hệ thống khỏi hiện tượng infinite loop, đảm bảo P95 Latency nằm trong tầm kiểm soát.
4. **Loại bỏ Validator Node:** Cắt giảm `validator_node` trung gian. `grader_node` sau khi kiểm tra chất lượng bản nháp chuyên môn sẽ đẩy thẳng luồng dữ liệu sang `conversation_node` để nạp Persona và stream về giao diện. Trách nhiệm xử lý dữ liệu rỗng hoặc fallback được tích hợp trực tiếp vào logic sinh của Synthesizer/Grader.
5. **Đồng bộ hóa Giao thức (SSE thay cho WebSocket):** Phê duyệt thay đổi sang SSE kết hợp REST POST theo tinh thần CDN-friendly, tự động reconnect và giảm tải hạ tầng mạng.

---

## 1. Kiến trúc Hệ thống Tổng thể (Hybrid Edge-Cloud)

```text
┌─── Cloud (VPS 2GB RAM + Supabase Managed) ────────────────────┐
│  CloudFront CDN                                              │
│  ├─ /static/* ──► AWS S3 (Cache 100% tài nguyên video/audio)│
│  └─ /chat/stream  ──► VPS Bypass (Chuyển tiếp SSE trực tiếp)  │
│                                                              │
│  FastAPI Gateway (VPS)                                       │
│  ├─ SSE Streaming & REST Endpoints                           │
│  ├─ LangGraph Agent Pipeline (Điều phối luồng gọi API Ngoài)  │
│  └─ Redis (Celery Broker + Short-Term Memory + Approval State)│
│                                                              │
│  Supabase (Managed)                                          │
│  └─ PostgreSQL + pgvector (Lịch sử hội thoại & Vector Search)│
├───────────────────────────────────────────────────────────────┤
│          ↕ Celery Task Queue qua Internet                     │
├─── Edge Worker (HP ProDesk, 48GB RAM, RTX 3060 12GB vRAM) ────┤
│  Celery Worker Cluster                                       │
│  ├─ Kimodo Task: GPU Inference ──► .mp4 ──► S3 Upload         │
│  └─ VieNeu-TTS Task: CPU Inference ──► .mp3 ──► S3 Upload    │
└───────────────────────────────────────────────────────────────┘
Chiến lược phân bổ tài nguyên phần cứng (Resource Slicing)
GPU (vRAM) - Độc quyền cho Motion: Card RTX 3060 12GB được biệt phái 100% dung lượng vRAM phục vụ mô hình khuếch tán hình ảnh NVIDIA Kimodo, xử lý chính xác các ràng buộc động học (kinematic constraints) cho vật lý trị liệu với độ trễ từ 5-10 giây.

CPU (RAM) - Độc quyền cho Speech: Mô hình VieNeu-TTS-v2-Turbo-GGUF chạy hoàn toàn trên CPU của máy HP ProDesk (48GB RAM). Giải phóng hoàn toàn GPU, hỗ trợ giọng đọc tiếng Việt mượt mà và code-switching Anh-Việt chuẩn xác.

2. Bản đồ luồng đi của Graph (Graph Flow)
Plaintext
START
  └─► memory_node (Truy xuất Redis STM + pgvector LTM)
        └─► planner_node (Lên phác đồ & Lựa chọn công cụ)
              │
              ├─ [SSE: approval_required] (Hỏi quyền chạy MCP Tools)
              ├─ [User Approves qua REST POST]
              │
              └─► retriever_node & executor_node (Chạy song song qua asyncio.gather)
                    │   [SSE: stage: executing_tool...]
                    │   [SSE: tool_output...]
                    │
                    └─► synthesizer_node (Tổng hợp kiến thức y khoa + kết quả tool)
                          │
                          └─► grader_node (Đánh giá chất lượng văn bản y khoa)
                                ├─ RETRY (Nếu lỗi & retry_count < 1) ──► planner_node
                                │
                                └─ PASS ───────────────────────────────► conversation_node (Persona Styling)
                                                                            └─► END
3. Bản Vá Lỗi Kỹ Thuật Hệ Thống (Technical Patch Matrix)
Sửa đổi triệt để 6 lỗi vận hành thực tế được phát hiện trong đợt rà soát mã nguồn:

Patch 1: Khắc phục Race Condition tại PostgreSQL Pool
Vấn đề: Khi memory_node gọi song song dữ liệu, tác vụ đọc profile không chứa vector codec chạy nhanh hơn sẽ chiếm quyền khởi tạo Pool trước, khiến các tác vụ chạy sau gọi pgvector bị sập hàng loạt.

Giải pháp: Áp dụng cơ chế Eager Initialization ngay tại lifespan khởi động của FastAPI. Bắt buộc khởi tạo kết nối database và đăng ký bộ giải mã vector (vector=True) từ Phase 0 trước khi server nhận request đầu tiên.

Patch 2: Tiếng Việt Có Dấu Cho Toàn Bộ Thông Báo Hệ Thống
Vấn đề: Các thông báo lỗi fallback viết không dấu gây suy giảm độ tin cậy của một ứng dụng HealthTech.

Giải pháp: Chuẩn hóa toàn bộ text hiển thị cho người dùng (User-facing text) trong grader_node và error_handler.py sang tiếng Việt chuẩn administrative (Ví dụ: "Xin lỗi, hệ thống đang gặp sự cố điều phối...").

Patch 3: Tối Ưu Token Tại reasoning_node
Vấn đề: Biến {query} bị nhồi hai lần vào cả System Prompt và User Message làm lãng phí token và gây hiện tượng over-anchor cho LLM.

Giải pháp: Tách biệt hoàn toàn template. System Prompt chỉ chứa cấu trúc luật phân tích và ngữ cảnh tài liệu; câu hỏi thô của user chỉ được đẩy duy nhất một lần vào tầng User Message.

Patch 4: Chuẩn Hóa Tem Thời Gian (Error Tracking)
Vấn đề: Toàn bộ trường timestamp trong log lỗi của các node đều để rỗng "", làm mất khả năng truy vết lỗi (Observability).

Giải pháp: Bắt buộc sử dụng hàm định dạng UTC ISO-8601 (datetime.now(timezone.utc).isoformat()) tại mọi điểm bắt lỗi errors.append.

Patch 5: Kích Hoạt Logic Ràng Buộc Khớp Xương (_extract_constraints)
Vấn đề: Hàm trích xuất góc khớp trả về mảng rỗng [] làm vô hiệu hóa thế mạnh kiểm soát động học của NVIDIA Kimodo.

Giải pháp: Triển khai hàm trích xuất regex / cấu trúc JSON nghiêm ngặt từ output của planner_node để bóc tách thông số khớp (Ví dụ: [{"joint": "right_shoulder", "angle": 90}]).

Patch 6: Đồng Bộ Logic Dự Phòng Node Manager
Vấn đề: Khi LLM classification bị lỗi, fallback gán confidence: 0.3 nhưng gán intent: knowledge_query, vi phạm quy định confidence < 0.5 -> clarify.

Giải pháp: Khóa cứng logic fallback của Manager Node: Nếu LLM lỗi, ép trạng thái về intent: clarify và định tuyến thẳng xuống conversation_node để hỏi lại thông tin một cách an toàn.

4. Chiến lược Phát dòng dữ liệu (SSE Event Schema)
Để xử lý khoảng thời gian chờ (5-10s) khi gọi các tool nặng song song, hệ thống liên tục bắn các sự kiện tiến độ về client:

TypeScript
type SSEEvent =
  | { event: "session_ready";    data: { session_id: string } }
  | { event: "stage";            data: { node: "planner" | "retriever" | "executor" | "synthesizer" | "grader" | "conversation", status: "started" | "complete" } }
  | { event: "approval_required"; data: { session_id: string; required_tools: string[] } }
  | { event: "system_status";    data: { message: "Đang tính toán động học và dựng hình 3D..." | "Đang truy xuất thư viện y khoa..." } }
  | { event: "tool_output";      data: { tool: string; result: any } } // Đẩy ngay kết quả thô của tool khi xong
  | { event: "token";            data: { content: string } } // Luồng văn bản cuối cùng từ Persona Agent
  | { event: "error";            data: { code: string; message: string } }
  | { event: "done";             data: {} }
5. Phân Định Kế Hoạch Triển Khai (Phase 1 Cắt Tải)
Mọi module được đóng gói độc lập để hai kỹ sư có thể lập trình song song mà không gây xung đột mã nguồn:

Task Developer N (Tầng Ứng Dụng AI & Tính Toán Nặng)
Thiết lập đồ thị tuần tự graph.py theo đúng cấu trúc hình phễu v2.3 (Bỏ Validator Node).

Viết logic bóc tách thông số góc khớp động học phục vụ module Kimodo.

Viết API Wrapper cho Kimodo (GPU) và VieNeu-TTS (CPU GGUF) bọc trong Docker Container chạy local.

Viết mã nguồn chấm điểm cho grader_node kèm giới hạn retry_count.

Task Architect T (Tầng Phân Phối & Quản Lý Luồng Giao Tiếp)
Triển khai Lifespan Eager Initialization để nạp bộ mã hóa pgvector cho Postgres Pool.

Thiết lập hệ thống endpoints FastAPI (POST /chat, GET /stream/{session_id}, POST /render/approve).

Thay thế cơ chế asyncio.Event bằng Redis Pub/Sub để quản lý trạng thái chờ phê duyệt (approval_required), đảm bảo hệ thống an toàn khi chạy đa luồng (Multi-process Uvicorn).

Xây dựng cơ chế Reconnect Recovery đọc tập hợp pending_tasks:{session_id} từ Redis khi client bị rớt mạng mạng hộ gia đình.