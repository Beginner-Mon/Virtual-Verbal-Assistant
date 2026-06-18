# FEATURE — YouTube paste-link Q&A (cụm B)

> Author: K | Date: 2026-06-13 | Audience: implementer (subagent) + N review sau
> Nguồn: TECH_DEBT 🟡 "User paste YouTube link". Owner approve build 13/06.
> Path: `agenticRAG/langgraph_agents/`. Tests: `pytest tests/langgraph_agents/`.

---

## Quyết định kiến trúc (K) — đọc trước

TECH_DEBT (29/05, TRƯỚC rebuild 3-axis) ghi "planner detect link". **Bỏ cách đó.**
Kiến trúc hiện tại: **transcript = 1 retrieval tool** mà retriever tự gọi (đúng D2b/D16
"manager giao WHAT, dev chọn tool"). Lợi:
- Tái dùng NGUYÊN đường evidence: ToolMessage → `synthesizer._extract_tool_results`
  → mode `synthesize`. KHÔNG state field mới, KHÔNG node mới, KHÔNG đổi routing.
- Empty/error đã có chuẩn D23 sẵn trong synthesizer + retriever.

**Phạm vi (chỉ build đúng đây — karpathy #2):** 1 tool mới + đăng ký + 1 đoạn prompt
+ tests. KHÔNG đụng planner/graph/state/synthesizer logic.

---

## Task 1 — Tool `youtube_transcript`

Thêm vào `tools/pgvector_tool.py` (cùng file các tool khác) HOẶC file mới
`tools/youtube_tool.py` (implementer chọn, ưu tiên cùng file cho gọn import).

```python
@tool
async def youtube_transcript(url: str) -> dict:
    """Fetch the spoken transcript of a YouTube video the user pasted.

    Use ONLY when the user's message contains a YouTube link (youtube.com/watch?v=
    or youtu.be/). Returns the transcript text (truncated) for answering questions
    about that specific video. Does NOT understand visuals — speech only.

    Args:
        url: The YouTube URL from the user's message (copy verbatim).

    Returns:
        {found: true, video_id, transcript, truncated: bool}  on success
        {found: false, reason: "no_transcript"}                if video has no captions
    """
```

Yêu cầu impl:
- Tái dùng `_extract_video_id` từ `youtube_ingest.py` (import, đừng copy). URL sai
  định dạng → `_extract_video_id` raise ValueError → bắt → trả `{found: false,
  reason: "invalid_url"}` (KHÔNG raise — invalid URL không phải service error).
- Fetch: `YouTubeTranscriptApi.get_transcript(video_id, languages=["vi","en"])` trong
  `asyncio.to_thread` (lib đồng bộ, đừng block event loop).
- Video không phụ đề → lib raise (TranscriptsDisabled/NoTranscriptFound) → bắt → trả
  `{found: false, reason: "no_transcript"}` (D23: empty, KHÔNG retry).
- Lỗi mạng/khác → raise (D23: service error → retriever retry/error_handler).
- **Truncate** (cap budget D28): nối `" ".join(e["text"])` → nếu > `_YT_CHAR_CAP`
  (≈ 12_000 chars ≈ 3k token) thì cắt head, set `truncated: true`. Hằng số module-level.
- KHÔNG embedding, KHÔNG ghi DB (KHÁC `ingest_youtube` — đó là đường admin/KB).
- Không cần `config`/scope (URL công khai, không đụng tenant).

## Task 2 — Đăng ký tool

`nodes/retriever_agent.py`: thêm `youtube_transcript` vào `RETRIEVER_BASE_TOOLS`
(graph `ToolNode` kế thừa tự động qua `RETRIEVER_BASE_TOOLS` — KHÔNG sửa graph.py).

## Task 3 — Prompt retriever

`_RETRIEVER_SYSTEM_PROMPT` mục TOOLS AVAILABLE + DECISION RULES: thêm 2-3 dòng:
- liệt kê `youtube_transcript(url)`;
- luật: "Nếu message user chứa link YouTube (youtube.com/watch hoặc youtu.be) → GỌI
  `youtube_transcript` với URL đó (copy nguyên văn). Đây là nguồn để trả lời về video."
- 1 example trong phần EXAMPLES nếu có chỗ.

> Tradeoff đã cân (karpathy #1): dựa vào LLM copy URL verbatim từ query. Model hiện đại
> làm tốt việc này. NẾU test/thực tế thấy LLM hay mangle URL dài → fallback: detect URL
> bằng regex trong planner, lưu state `youtube_url`, retriever force-call. KHÔNG làm
> trước khi có bằng chứng flaky (đừng over-build).

## Task 4 — Tests (`tests/langgraph_agents/test_youtube_paste.py`)

Mock `YouTubeTranscriptApi.get_transcript` — KHÔNG gọi mạng thật trong test.
- [ ] `_extract_video_id` qua tool: `watch?v=ABC` và `youtu.be/ABC` → cùng video_id
- [ ] URL không hợp lệ ("https://example.com") → `{found: false, reason: "invalid_url"}`, KHÔNG raise
- [ ] transcript ngắn → `{found: true, transcript, truncated: false}`
- [ ] transcript dài (> cap) → `truncated: true`, len ≤ cap
- [ ] video không phụ đề (mock raise NoTranscriptFound) → `{found: false, reason: "no_transcript"}`, KHÔNG raise
- [ ] lỗi mạng (mock raise generic Exception) → tool RAISE (để retriever xử D23)
- [ ] `youtube_transcript` có trong `RETRIEVER_BASE_TOOLS`
- [ ] tool schema chỉ expose `url` (LLM-visible), không có field thừa

## Định nghĩa XONG
1. Tool đúng spec; reuse `_extract_video_id`; truncate có cap; empty≠error đúng D23
2. Đăng ký + prompt cập nhật; graph.py KHÔNG đổi
3. Tests trên pass; full suite xanh (225 + mới), 0 regression
4. KHÔNG đụng planner/synthesizer/state/routing
5. Worklog → K review

## Ngoài phạm vi (đừng làm)
- Semantic-chunk transcript theo câu hỏi (head-truncate là đủ MVP; ghi nợ nếu muốn).
- Whisper STT cho video không phụ đề.
- Ghi transcript vào KB / LTM riêng (LTM nhớ tự nhiên qua summarizer M.5 — câu trả lời
  đã chứa nội dung video).
- Planner/regex detection (chỉ làm nếu LLM-copy-URL chứng minh flaky).
