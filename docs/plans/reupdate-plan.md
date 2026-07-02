# VVA — Re-Architecture Plan

> Architect: K | Cập nhật: 2026-06-02
> Audience: N (Developer), T (Reviewer), Owner
> Scope: file kế hoạch sửa đổi so với `plans/v2.4-plan.md`. plans/v2.4-plan KHÔNG sửa cho tới khi rebuild xong.

---

## Context

Tài liệu này tổng hợp quyết định kiến trúc qua các session 26/05 → 02/06/2026.
**Quyết định lớn (02/06): rebuild Memory + Intent layer** trên nhánh `feature/langgraph-rewrite`
(làm trực tiếp, không nhánh con). Phần "Memory & Intent Rebuild" (§M) bên dưới là **nguồn chân lý
duy nhất** — gồm cả spec lẫn giải thích (đã gộp file `planner-refactor-explained.md` vào §M.intuition,
file đó đã xóa). Nó **thay thế** `architecture/schema-redesign.md §3-6` và phần 6.10.1 (normalize messages) cũ.

> ⚠️ Mục "Phase 6.10 — Pre-deploy Hardening" phía sau phần lớn ĐÃ XONG (worklog 27-28/05) hoặc
> bị **superseded** bởi rebuild bên dưới (đặc biệt 6.10.1 DB normalize → thay bằng schema mới §M.1).
> Giữ lại để tham chiếu lịch sử, không phải spec hiện hành.

---

# 🔧 MEMORY & INTENT REBUILD (02/06 — nguồn chân lý mới)

> Build trực tiếp trên `feature/langgraph-rewrite`. Bắt đầu từ memory layer, học từ sai lầm
> kiến trúc cũ (bảng `embeddings` polymorphic bê từ ChromaDB → leak chéo user, no FK, A1-A6).
> Thứ tự: schema memory → intent refactor (3 trục) → grader tag-map → retriever flags.
>
> **Phạm vi đã chốt: chỉ MEMORY + PLANNER.** Layer khác (retriever impl, synthesizer, deploy)
> chưa redesign trong vòng này.

## M.intuition — Trực giác (đọc trước khi vào spec)

**Metaphor xuyên suốt** (dùng giải thích cho team):
- **Planner = manager** — ra việc cần bàn giao (WHAT), KHÔNG chỉ định cách làm
- **Retriever = dev** — tự chọn tool (KB/web/memory) + tự suy chi tiết tìm kiếm (HOW)
- **Synthesizer = phòng tổng hợp** — viết câu trả lời cuối
- **Grader = QA** — check deliverable đủ chưa

**LLM stateless, context window = "tờ note":** LLM không nhớ gì giữa các lượt — mỗi request đọc 1 tờ
note sạch (system + user_facts + summary + recent raw + query) rồi quên. "Nhớ" = memory node **viết**
lịch sử lên tờ note trước khi đưa LLM đọc. Cả memory layer = nghệ thuật *viết gì lên tờ note có hạn
chỗ*. → đây là lý do `resolved_query` cần memory chạy TRƯỚC planner (để resolve "nó/cái đó").

**3 trục PlanOutput độc lập — bảng 4 tổ hợp** (không trục nào derive từ trục khác):

| Query | needs_retrieval | required_outputs |
|-------|:---:|---|
| "xin chào" | ❌ | `[]` |
| "tôi đau ngực khi tập" | ❌ | `[red_flag_screen]` ← có tag, KHÔNG tra cứu |
| "giá vàng hôm nay?" | ✅ | `[]` ← tra cứu, KHÔNG tag |
| "bài tập cho L4-L5" | ✅ | `[exercise_protocol]` |

→ Cả 4 ô tồn tại → `needs_retrieval` và `required_outputs` KHÔNG suy ra nhau → 2 trục riêng, 2 cổng
routing độc lập (D15). Gộp = lỗi an toàn (xem M.2).

**`resolved_query` vs `required_outputs` — đừng lẫn:**
- `resolved_query` = **user HỎI gì** (input, 1 câu, synthesizer luôn đọc) — như *đề bài*.
- `required_outputs` = **trả lời PHẢI CÓ gì** (output checklist, tag, grader đọc) — như *rubric chấm*.
- resolved_query dùng *trước* (hiểu hỏi gì); required_outputs dùng *sau* (chấm trả lời).

## M.0 — Quyết định đã chốt (decision log)

| # | Quyết định | Lý do |
|---|-----------|-------|
| D1 | **Intent 6-enum → 3 trục** (required_outputs[str] / resolved_query / routing bits) | enum ép chọn 1; tách deliverable–query–routing thành 3 trục độc lập |
| D2 | **Routing ⟸ needs_retrieval; grader ⟸ required_outputs** (2 cổng độc lập) | tách trục, không đọc nhầm nhau |
| D2b | **Manager giao WHAT, dev chọn tool**: bỏ `needs_kb/web/memory`, retriever tự chọn + tự suy chi tiết | chỉ định tool = việc dev; planner ra WHAT (tag+query), không HOW (tool nào, tìm chi tiết gì) |
| D3 | **needs_motion = HARD GATE** (edge cứng, không để LLM/MCP-description tự đoán) | GPU 5-10s; action không phải retrieval; cost bất đối xứng — không cược vào LLM đoán |
| D4 | **is_clinical BỎ** — persona lấy từ session config (UI chọn) | framing không phải per-turn decision |
| D5 | **required_outputs = controlled-vocab đóng** (không freeform) | grader rule-based cần nhãn cố định để áp rule deterministic |
| D6 | **Tag 2 loại: safety → template cứng (no retry); quality → retry** | an toàn không được phụ thuộc LLM retry may rủi |
| D7 | **TAG_RULES = 1 dict nguồn-chân-lý** + startup assertion | chống drift tag↔rule |
| D8 | **required_outputs=[] → grader skip** | general/chat/clarify gộp tự nhiên (không tag = không contract) |
| D9 | **expanded_query → resolved_query** (coreference resolve ở planner) | memory chạy TRƯỚC planner → planner có đủ context resolve "đào sâu về nó" |
| D10 | **Embedding: e5-small `vector(384)`** mọi bảng + **prefix bắt buộc** (`query:`/`passage:`) | mạnh hơn MiniLM cho tiếng Việt, cùng dim (free upgrade) |
| D11 | **Memory tách Knowledge** — 2 index/tool riêng | private per-user vs public KB; gốc của leak cũ |
| D12 | **GDPR deletion cascade — thiết kế TỪ ĐẦU** | đụng tính "đóng băng" của summary, không defer được |
| D13 | **Summary: 1 ngưỡng = 10k tokens** (recent-raw window = chunk_size) | recent raw là phần UNCACHED = driver chi phí; "2 ngưỡng" tạo vùng giữa vô chủ |
| D14 | **user_memory gom profile**; user tự ghi (MVP), AI-auto sau | đơn giản trước; `valid` flag sẵn cho conflict khi thêm AI-auto |
| D15 | **2 cổng độc lập**: retriever⟸needs_retrieval, grader⟸required_outputs (KHÔNG gộp) | gộp = bug an toàn (no-retrieval + safety tag → skip grader → template không ép) |
| D16 | **memory_search = TOOL của retriever** (dev tự chọn) | retriever suy từ tag+query → chọn kb/web/memory; kb+memory gọi song song |
| D18 | ~~required_outputs = list[{tag, scope}]~~ → **list[str] (tag thuần)** (điều chỉnh 02/06) | scope overlap resolved_query (thừa); retriever LLM tự suy chi tiết từ query+tag+STM; list[str] gọn hơn. Khôi phục {tag,scope} nếu sau cần chỉ dẫn chặt — xem M.1 |
| D17 | **GDPR mark-dirty = app logic** + cột `status` + hard-delete + xóa empty-chunk | summaries không FK→messages; cửa sổ dirty phải loại chunk khỏi context+search |
| D19 | **Bỏ `user_id` khỏi messages + summaries** (chỉ giữ session_id) | session_id→user là đủ; hot path dùng session_id; GDPR qua cascade; memory_search scope 2-step. Tenant-scope tập trung ở ít tool function (không rải rác), LLM sinh args không sinh SQL → "dễ quên" không thực, RLS chưa cần (một-cửa). 3NF sạch. Revisit khi nhiều đường vào DB → RLS (Phase 7) |
| D20 | **Retriever KHÔNG nhận memory_context; retrieval ≠ personalize** | planner đã distill vào resolved_query; retrieval lấy sự-thật (giống mọi user), personalize ở synthesis, an toàn qua tag — 3 cơ chế tách bạch |
| D21 | **resolved_query giữ tool-selection cue** (temporal/source/topic), chỉ bỏ đại từ | bỏ memory_context khỏi retriever → resolved_query phải tự đủ cho retriever chọn tool |
| D22 | **Clarify 2 loại: tĩnh (planner) + động (tool ambiguous → synthesizer)**; multi-turn không loop | clarify động chỉ biết sau khi query DB; hỏi→END→turn mới, không node/loop/field mới |
| D23 | **Tool phân biệt empty `{found:[]}` vs error `{error}`** | empty→synthesizer, error→retry→error_handler; lẫn nhau = nói "không có" khi service sập |
| D24 | **Retry CHỈ cho service error, KHÔNG cho empty** | plain-retry empty vô ích (query không đổi → kết quả không đổi); retry-rewrite là M.11 defer |
| D25 | **Out-of-scope = tag `referral_advice`, KHÔNG severity enum**; clinical-no-source → refuse không bịa | free/caution/refuse emerge từ tag (no-tag/scope_disclaimer/referral_advice); AI tự chấm severity y tế = vượt vai advisory |
| D26 | **#1 motion = node riêng song song retriever** (Kimodo MCP gọi từ edge cứng, planner-gated) | motion=action≠retrieval; tách khỏi retriever để: đúng D3 hard-gate, retry không re-fire GPU, song song thật. MCP=transport, edge=control (2 tầng độc lập) |
| D27 | **web-off + cần real-time = instance của no-source** (D25), không case riêng | user toggle off → web loại khỏi tools → mọi tool rỗng → no-source path + cue "web user-off" để synthesizer giải thích |
| D28 | **Tool-result: cap top-k (5/3/3), KHÔNG cơ chế chống tràn** | định lượng: ~3.8k tool + ~13k note ≈ 17k << 64k → không tràn happy path; rerank/truncate là over-eng. Nợ: đo lại nếu đổi provider/chunk |
| D29 | **Synthesizer mode EMERGE từ signals, KHÔNG enum `response_mode`** | response_mode = intent đội tên khác; refuse/clarify-động chỉ biết SAU tool → derive từ (clarification + tool state + tags), không store |
| D30 | **Persona áp MỌI mode (gồm refuse/clarify); streaming B (cảnh báo append cuối)** | persona=giọng user chọn, không đổi nội dung; an toàn = grader-net + persona-prompt, không cấm persona. Stream + disclaimer append cuối |
| D31 | **Grader rule-based = lưới chặn "quên hẳn", KHÔNG bắt cường độ** | regex bắt vắng-mặt-marker, không hiểu ngữ nghĩa; cường độ cảnh báo = persona/synthesizer prompt ép format (phương án D). Advisory nên chấp nhận. Đường lên LLM-judge (soft→hard) khi có data. Grader bàn lại sau |
| D32 | **2 loại cảnh báo tách biệt: safety=ĐẦU (synthesizer), unverified disclaimer=CUỐI (grader append)** | sửa D30: safety warning phải đọc trước khi làm → đầu output mọi persona; disclaimer "chưa kiểm chứng" = boilerplate cuối. Khác mục đích, khác vị trí |
| D33 | **Danger lộ trong query → planner detect (1 nơi); KHÔNG thêm field safety_note** | red-flag PT lộ trong câu hỏi ("đau ngực"); planner emit referral_advice/red_flag_screen → synthesizer thừa hành (viết cảnh báo đầu). Không 2-nơi-phán. `required_outputs` giữ list[str]. Grader-fail-safety → fallback template-đầu PER-TAG (gỡ #1, không cần detail động) |
| ~~D26~~ | **REVISE: motion độc lập, synthesizer KHÔNG nhận flag** (bỏ option C) | coherence qua tag `motion_descriptor` (synthesizer viết mô tả) + UI ghép text+video cùng chủ đề; flag báo motion = thừa (trùng tag). Vẫn: Kimodo node riêng, MCP từ edge, planner-gated |

## M.1 — Intent → 3 trục (thay 6-enum)

> Triết lý: **Planner = manager** ra việc cần bàn giao (WHAT). **Retriever = dev** tự chọn tool (HOW).
> Planner KHÔNG chỉ định tool cụ thể (kb/web/memory) — đó là việc dev.

```python
class PlanOutput:
    # TRỤC 1 — required_outputs: DELIVERABLE (manager giao việc — tag thuần)
    #   tag = hạng mục bàn giao (enum đóng ∈ TAG_RULES) → GRADER đọc, map rule
    required_outputs: list[str]     # ["exercise_protocol", "red_flag_screen", ...]

    # TRỤC 2 — resolved_query: coreference đã resolve (D9). GIỮ SẠCH (chỉ là câu hỏi).
    resolved_query: str             # "đào sâu về nó" → "chi tiết bài bird-dog"

    # TRỤC 3 — 2 routing bit (manager quyết, KHÔNG chỉ định tool nào)
    needs_retrieval: bool           # "có cần tra cứu không" → retriever vs synthesizer thẳng
    needs_motion: bool              # Kimodo GPU — HARD GATE (D3); action, không phải retrieval
```

**Retriever tự suy "tìm gì" từ 3 mảnh** (không cần planner pre-compute scope):
`resolved_query` (câu hỏi) + `required_outputs` (tag cần gì) + STM/user_facts (ngữ cảnh: "65 tuổi,
đau mãn") → LLM tự suy "tìm bài NHẸ cho L4-L5 hợp người 65 tuổi". Đúng metaphor: dev tự lo HOW.

> 📌 **ĐIỀU CHỈNH (02/06): bỏ `scope`.** Bản trước có `required_outputs: list[{tag, scope}]` — scope
> là text chỉ dẫn tìm kiếm per-tag. Bỏ vì: (1) scope overlap resolved_query (dữ liệu thừa — teammate
> bắt đúng qua ví dụ L4-L5); (2) retriever là LLM, tự suy chi tiết từ query+tag+STM được, không cần
> manager pre-compute; (3) `list[str]` đơn giản hơn `list[dict]`, grader đọc tag trực tiếp.
> **KHÔNG nhồi scope vào resolved_query** (giữ query sạch — nhồi chỉ dẫn vào query làm nhiễu embedding).
> **Đánh đổi**: mất tính deterministic của scope (planner không ghim chỉ dẫn) → giao retriever LLM tự
> suy, kém kiểm soát chút với multi-deliverable phân kỳ; bù lại grader vẫn chặn ở sau. Nếu sau này cần
> chỉ dẫn chặt cho case phức tạp → khôi phục `{tag, scope}` (lý do còn đây).

**Vì sao `needs_retrieval` 1 bit thay 3 flags (kb/web/memory):**
- 3 flags = manager chỉ định tool = lấn vai dev (vi phạm separation of concerns).
- 1 bit = manager chỉ nói "cần tra cứu không", dev (retriever) tự chọn KB/web/memory.
- KHÔNG derive được từ `required_outputs != []`: case "đau ngực" có tag `red_flag_screen` (cần grader)
  nhưng KHÔNG cần retrieval → 2 cổng phải độc lập (D15), nên retriever cần bit riêng.

**Bỏ khỏi PlanOutput**: `intent`, `expanded_query`, `scope` (xem điều chỉnh trên), `search_strategy`,
`constraints_detected`, `is_clinical`, `notes`, `confidence`, 3 flag `needs_kb/web/memory`.
Persona KHÔNG ở đây (session config từ UI).

## M.2 — Routing (graph edges) — HAI CỔNG ĐỘC LẬP

> ⚠️ Cổng retriever và cổng grader đọc **2 trục khác nhau**. KHÔNG gộp điều kiện.
> Gộp = bug an toàn: query không cần tra cứu nhưng có safety tag (vd "đau ngực khi tập" →
> `needs_retrieval=false` NHƯNG required_outputs=[{tag:red_flag_screen}]) sẽ skip grader →
> template an toàn không được ép → LỖ HỔNG.

```
memory (STM + user_memory facts, TRƯỚC planner)
  → planner (phát required_outputs + resolved_query + needs_retrieval + needs_motion)
       │
       ├─ Cổng RETRIEVER  ⟸ needs_retrieval?          (đọc bit retrieval)
       │     true   → retriever (DEV tự chọn kb/web/memory, gọi song song) → synthesizer
       │     false  → synthesizer (skip retriever)
       │
       └─ (sau synthesizer) Cổng GRADER ⟸ required_outputs != []?  (đọc REQUIRED_OUTPUTS)
             != []  → grader (safety: template cứng / quality: retry) → END
             == []  → END   (fast-path: chat/general/clarify không tag)

  needs_motion=true → HARD edge force Kimodo (không qua LLM tool-choice)
```

**Retriever input** (D20): CHỈ `resolved_query` + `required_outputs` (+ `grader_feedback` khi retry).
KHÔNG nhận memory_context/STM. Lý do: planner đã distill ngữ cảnh vào resolved_query rồi; đưa lại
raw facts = double-handling, fast-model retriever tự personalize lệch ý planner.

**Nguyên tắc retrieval ≠ personalize** (D20): retrieval lấy **sự thật** (bài tập bird-dog làm sao —
giống mọi user); personalize ("nói cho người 65 tuổi") là việc **synthesis**; an toàn (chống chỉ định)
đi qua **tag** (`contraindication`/`red_flag_screen`), KHÔNG qua retrieval personalize. 3 cơ chế tách bạch.

**Retriever chọn tool thế nào** (DEV tự quyết, D16 song song):
- đọc `resolved_query` + `required_outputs` (tag) → suy ra cần gì:
  - chủ đề PT/wellness → `kb_search`
  - thông tin mới/real-time/giá/tin tức → `web_search`
  - nhắc quá khứ user ("lần trước", "tôi đã nói") → `memory_search` / `resume_last_session`
- gọi nhiều tool SONG SONG nếu cần (vừa KB vừa web).

**Ràng buộc planner** (D21): khi resolve, GIỮ tool-selection cue trong resolved_query — temporal
("tuần trước"), source ("mới nhất"), topic — chỉ bỏ đại từ mơ hồ ("nó/cái đó"). Lược mất cue →
retriever mất tín hiệu chọn tool.

**Mấu chốt**: kể cả `needs_retrieval=false` vẫn QUA grader NẾU có tag (bịt lỗ safety).
Fast-path chat (`needs_retrieval=false` + tags rỗng) vẫn đi thẳng END (giữ tốc độ).
Cổng retriever ⟸ `needs_retrieval`; Cổng grader ⟸ `required_outputs`. Không cùng điều kiện.

## M.2b — Clarify: TĨNH vs ĐỘNG (D22)

Có **2 loại clarify khác bản chất** — plan cũ chỉ xử loại tĩnh:

| | Clarify TĨNH | Clarify ĐỘNG |
|---|---|---|
| Biết cần hỏi khi nào | **trước** retrieval | **chỉ sau** khi query DB |
| Ví dụ | "bài tập" (thiếu vùng đau) | "session vừa rồi" + DB có 5 session gần |
| Planner tự biết? | ✅ có | ❌ không — phải query mới biết ambiguous |

**Cơ chế thống nhất: clarify = multi-turn, KHÔNG loop trong graph.** Hỏi user → END. User trả lời =
turn mới, đi lại graph từ đầu (planner giờ có thêm context → resolve chặt hơn). Không node clarify,
không loop, không planner-đoán-trước.

```
Tĩnh:  planner detect thiếu info → needs_retrieval=false → synthesizer HỎI → END
Động:  planner KHÔNG đoán ambiguous → retriever gọi tool → tool trả {ambiguous, candidates}
       → synthesizer thấy evidence mơ hồ → HỎI "session nào: (1)... (2)..." → END
       → turn sau user chọn → planner resolve chặt → chạy bình thường
```

**Cơ chế cần thêm (nhỏ):**
1. Tool memory_search/resume trả **ambiguity metadata** trong ToolMessage (`found N`, `ambiguous: bool`,
   `candidates: [...]`).
2. Synthesizer prompt thêm 1 luật: *"nếu evidence mơ hồ / nhiều lựa chọn ngang nhau → HỎI user làm rõ,
   ĐỪNG đoán bừa."*

Không field state mới, không node mới, không loop. Synthesizer vốn đã đọc tool results → "thấy mơ hồ
thì hỏi" là hành vi tự nhiên. Synthesizer hỏi khi: (planner flag clarify tĩnh) HOẶC (tool trả ambiguous).

## M.2c — Error / Empty / Out-of-scope (retriever + grader hành vi biên)

**3 trạng thái tool — PHẢI phân biệt** (D23):
```
{found: [...]}    kết quả OK
{found: []}       tool chạy OK, 0 kết quả  → KHÔNG retry (cùng query → cùng rỗng), đi synthesizer
{error: "..."}    service chết (timeout/crash) → RETRY (có thể hồi) → vẫn lỗi → error_handler
```
Lẫn empty với error = nói "không có bài tập" trong khi service sập = sai + nguy hiểm. Tool trả
ToolMessage rõ: `"Không tìm thấy kết quả cho [query]"` (empty) vs raise error (service).
**Retry CHỈ cho service error, KHÔNG cho empty** (D24 — plain-retry empty vô ích vì deterministic).
*(retry-with-rewritten-query là M.11 query-rewrite, đã defer — khác chuyện.)*

**Empty / no-source → hành vi tùy ca** (D25):
| Ca | Điều kiện | Hành vi |
|---|---|---|
| general no-KB nhưng web có | general query, web trả kết quả | LLM trả theo web + citation (`evidence_citation`) + cảnh báo nhẹ "theo web, có thể đổi" |
| clinical CÓ nguồn | có ToolMessage non-empty | trả lời + `scope_disclaimer` |
| **clinical KHÔNG nguồn nào** | tag clinical + mọi tool rỗng | **KHÔNG bịa** → `referral_advice`: "không tìm thấy thông tin đáng tin, hãy hỏi chuyên gia" |

**Out-of-scope = path refuse, KHÔNG phải severity enum** (D25):
3 "mức" (free/caution/refuse) KHÔNG cần field mới — chúng EMERGE từ tag nào planner phát:
- **Answer freely** = no tag (`required_outputs=[]`)
- **Answer with caution** = `scope_disclaimer`
- **Out of scope** = `referral_advice` + `needs_retrieval=false`

> ⚠️ KHÔNG thêm trục `severity`: trùng tag (1 fact 2 nơi); và "AI tự chấm mức nghiêm trọng y tế"
> = vượt vai advisory (ta KHÔNG có y khoa chính quy — không giả vờ triage lâm sàng).

**Ranh giới scope advisory** (planner học để phát `referral_advice` khi vượt):

| TRONG scope (trả lời) | OUT of scope (refuse + refer) |
|---|---|
| bài tập, tư thế, stretch, wellness chung | chẩn đoán bệnh ("tôi bị gì?") |
| giải phẫu cơ bản | kê thuốc / liều |
| khi nào nên đi khám (red flag) | diễn giải xét nghiệm / ảnh y tế |
| | tình trạng cấp tính / hậu phẫu nặng |

→ clinical-no-source (V2 ca b) rơi vào out-of-scope → `referral_advice` refuse, không chế bài tập.

**User tắt web + query cần real-time = instance của no-source** (D27, không phải case riêng):
web bị user toggle off → web_search loại khỏi tools (user override > LLM choice) → mọi tool rỗng →
no-source path. Synthesizer giải thích đúng lý do: "không có thông tin real-time (web search đang tắt),
bật để tôi tìm." KHÔNG cần message đặc biệt từ retriever — tái dùng no-source, thêm cue "web user-off".

**Tool-result budget (D28)** — KHÔNG tràn ở happy path, chỉ cần cap top-k tỉnh táo:
- default: `kb top-k=5` (~2.5k tok), `memory top-k=3` (summary đã nén ~0.9k), `web top-k=3` (~0.45k)
  → tool results ~3.8k; cộng "tờ note" (~12-13k) ≈ 16-17k, an toàn với context 64k+.
- Đây là **default an toàn**, KHÔNG phải cơ chế chống tràn (thực tế không tràn). Không rerank/truncate.
- **Nợ (revisit, không task)**: đổi sang provider context nhỏ HOẶC chunk e5 to hơn → đo lại budget.

## M.3 — Grader (tag-driven, deterministic)

```python
# 1 nguồn chân lý (D7). tag → (kind, rule_fn). kind ∈ {safety, quality}.
TAG_RULES = {
    # ── SAFETY (thiếu → chèn TEMPLATE CỨNG, KHÔNG retry — D6) ──
    "red_flag_screen":   ("safety", has_danger_warning),   # cảnh báo dấu hiệu nguy hiểm (đau ngực, tê, mất kiểm soát...)
    "referral_advice":   ("safety", has_referral),         # khuyên đi gặp bác sĩ/chuyên gia khi vượt scope wellness
    "scope_disclaimer":  ("safety", has_disclaimer),       # "đây là tư vấn wellness, không thay khám lâm sàng"

    # ── QUALITY (thiếu → retry max 1 — D6) ──
    "exercise_protocol": ("quality", has_sets_reps_frequency),  # bài tập phải có sets + reps + tần suất
    "exercise_steps":    ("quality", has_ordered_steps),       # hướng dẫn thực hiện ≥2 bước
    "contraindication":  ("quality", has_contraindication),    # nêu trường hợp KHÔNG nên tập
    "evidence_citation": ("quality", has_source),             # knowledge query phải có nguồn/citation
    "motion_descriptor": ("quality", has_motion_fields),      # visualize_motion: mô tả động tác + khớp
}
# startup: assert planner_tags ⊆ set(TAG_RULES)   # fail-fast chống drift (D7)

for tag in required_outputs:                # required_outputs = list[str] (tag thuần)
    kind, rule = TAG_RULES[tag]
    if not rule(answer):
        if kind == "safety":   chèn TEMPLATE CỨNG (deterministic, NO retry)   # D6
        if kind == "quality":  retry (max 1)                                  # D6
if required_outputs == []:   skip toàn bộ grader                              # D8
```

**Bộ tag PT domain (8 tag):**

| Tag | Kind | Rule (deterministic check) | Khi planner phát |
|-----|------|---------------------------|------------------|
| `red_flag_screen`   | safety | có cảnh báo dấu hiệu nguy hiểm | triệu chứng đáng ngờ (đau ngực, tê, chóng mặt) |
| `referral_advice`   | safety | có khuyên đi khám | vượt scope wellness, cần chuyên gia |
| `scope_disclaimer`  | safety | có disclaimer wellness | mọi câu tư vấn lâm sàng |
| `exercise_protocol` | quality | có sets + reps + tần suất | recommend bài tập cụ thể |
| `exercise_steps`    | quality | có ≥2 bước thứ tự | hướng dẫn thực hiện động tác |
| `contraindication`  | quality | nêu trường hợp tránh | bài tập có rủi ro với 1 số tình trạng |
| `evidence_citation` | quality | có nguồn/citation | knowledge query y khoa |
| `motion_descriptor` | quality | có mô tả động tác + khớp | visualize_motion (đi kèm Kimodo) |

> Rule check là **heuristic marker-based** (regex/keyword), không LLM — giữ determinism + testable.
> Bộ này là baseline; thêm tag = thêm 1 dòng vào TAG_RULES + cập nhật planner prompt vocab (assert
> chống quên). KHÔNG cho planner chế tag ngoài danh sách.

KHÔNG dùng LLM-judge làm cổng cứng (mất determinism). LLM-judge nếu cần = lớp soft advisory.

**Tầm của grader rule-based (D31) — thành thật về giới hạn:**
- Bắt được: **vắng mặt marker** ("quên hẳn" cảnh báo → has_referral=false → fail). Ca tệ nhất.
- KHÔNG bắt được: **cường độ** ("có nhắc bác sĩ nhưng quá nhẹ" → regex thấy "bác sĩ" → pass). Regex
  không hiểu ngữ nghĩa.
- → Phân lớp: grader = lưới thô chặn "quên hẳn"; **cường độ cảnh báo = việc PERSONA PROMPT** (dạy
  mọi persona: ca red-flag phải nói rõ "ngừng/đi khám"). Không over-claim grader là thẩm phán cường độ.
- Chấp nhận được vì hệ thống là **advisory wellness** (không lâm sàng). Grader optional-ish nhưng nên
  giữ (bỏ hẳn = mất lưới "quên hẳn"). **Bàn lại sau** (Mr. N để ngỏ).

## M.3b — Synthesizer (universal responder, persona-styled)

Node DUY NHẤT viết cho user. Đọc state → suy mode → viết (stream) → final_answer.

**Danger detection = PLANNER, không phải synthesizer** (D33): red-flag PT lộ trong query
("đau ngực khi tập") → planner phát `red_flag_screen`/`referral_advice`. "ngồi nhiều đau lưng, bài
tập tại nhà?" → lành tính, planner KHÔNG phát safety tag. Synthesizer chỉ THỪA HÀNH tag (viết cảnh
báo đầu), KHÔNG tự đánh giá lại danger (tránh 2-nơi-phán lệch nhau). Không cần field `safety_note`.

**Mode EMERGE từ signals, KHÔNG enum `response_mode`** (D29 — derive không store, như cả plan):
```
needs_clarification OR tool trả ambiguous   → CLARIFY  (hỏi lại; multi-turn, M.2b)
clinical tag + mọi tool rỗng (no-source)    → REFUSE   (referral_advice, M.2c — không bịa)
có ToolMessage non-empty                    → SYNTHESIZE (tổng hợp + citation từ ToolMessage)
không tool + không tag                      → CHAT     (greeting/general)
```
KHÔNG dùng `intent`/`plan`/`constraints_detected`/`notes` (code cũ — đã chết). Suy từ:
`needs_clarification` + tool state (messages) + `required_outputs` tags.

**Persona áp MỌI mode, gồm refuse/clarify** (D30): persona = giọng (user chọn ở UI, session config),
KHÔNG đổi nội dung bắt buộc. Persona viết tự do câu cảnh báo/refuse theo giọng → **grader check sau**
(safety tag) bù template nếu thiếu marker. Persona mặc định = giọng chuyên nghiệp nhất.
→ An toàn = grader-net (vắng mặt) + persona-prompt (cường độ), KHÔNG phải cấm persona.

**Streaming (D30, chọn B) + 2 loại cảnh báo TÁCH BIỆT (D32)**:
- Stream token ra SSE khi gen.
- **Safety warning** (red_flag / refuse — nguy hiểm sức khỏe): **ĐẦU output**, synthesizer ép viết
  mở đầu (mọi persona, không ngoại lệ). Phải đọc TRƯỚC khi làm theo → không thể để cuối (user ngừng
  đọc giữa chừng = bỏ lỡ). Grader chỉ VERIFY có mặt (rule marker), không append.
- **Unverified disclaimer** (no-KB / không nguồn uy tín — thông tin chưa kiểm chứng y tế): **CUỐI
  output**, grader append. Boilerplate, không khẩn.
- → KHÔNG trộn: safety=đầu (synthesizer viết), disclaimer=cuối (grader append). Khác mục đích, khác chỗ.

**Motion (revise D26 — bỏ C)**: synthesizer KHÔNG nhận motion flag, KHÔNG biết video gen hay chưa.
Coherence qua tag `motion_descriptor` (planner phát khi needs_motion) → synthesizer viết MÔ TẢ động
tác. Kimodo node đẩy video thẳng UI độc lập. UI ghép text+video (cùng chủ đề → tự khớp). "Việc ai nấy
làm." Đi sâu motion-text coupling = nợ tương lai.

**Output shape:**
```python
{
    "final_answer": "...",   # persona-styled, đã stream ra UI; bản đầy đủ cho grader + ghi messages
    "total_tokens": ...,
}
# KHÔNG ghi: reasoning_output (bỏ rồi), motion (node riêng). Grader đọc thẳng final_answer.
```

## M.4 — Schema (Postgres + pgvector, e5-small 384)

```sql
CREATE EXTENSION IF NOT EXISTS vector;   -- note: bật trước HNSW

CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    auth_provider TEXT, auth_subject TEXT,        -- Phase 7 auth; dev/anon dùng uuid5
    display_name  TEXT,
    created_at    TIMESTAMPTZ DEFAULT now(), updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (auth_provider, auth_subject)
);

CREATE TABLE conversations (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT,                              -- nullable; UI fallback first-msg preview
    created_at  TIMESTAMPTZ DEFAULT now(), updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE messages (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seq_id       BIGSERIAL,                        -- thứ tự GLOBAL, có gap, luôn kèm WHERE session_id
    session_id   UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content      TEXT NOT NULL,
    token_count  INTEGER,                          -- tính LÚC GHI bằng tokenizer LLM; null → ngưỡng hỏng
    created_at   TIMESTAMPTZ DEFAULT now()
);
-- KHÔNG có user_id (D19): session_id → conversations.user_id là đủ. Hot path (assemble/paginate/
-- summarize) dùng session_id; GDPR xóa user qua cascade; tenant-scope ở tool function. 3NF sạch.

CREATE TABLE summaries (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    summary_text     TEXT NOT NULL,
    covers_from_seq  BIGINT NOT NULL,
    covers_up_to_seq BIGINT NOT NULL,
    embedding        vector(384),                  -- điền NGAY khi tạo (1 lần), không để null
    status           TEXT NOT NULL DEFAULT 'active',-- 'active' | 'dirty' (GDPR, M.8 #1)
    created_at       TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_chunk UNIQUE (session_id, covers_from_seq)   -- idempotent CAS
);
-- KHÔNG có user_id (D19): memory_search scope tenant bằng 2-step (session_ids của user → ANY).
-- Xem M.6. Revisit khi có nhiều đường vào DB ngoài tool (analytics/admin/RLS) — Phase 7.
-- 'dirty' = chunk chứa message vừa bị xóa, đang chờ re-summarize nền.
-- Chunk 'dirty' bị LOẠI khỏi cả context assembly LẪN memory_search (dùng raw còn lại tạm thời).

CREATE TABLE user_memory (                          -- D14: gom profile; fact bền vững luôn-bật
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fact_text   TEXT NOT NULL,
    category    TEXT,
    valid       BOOLEAN DEFAULT true,               -- fact cũ bị thay → false (conflict)
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE documents (                            -- KB nguồn (Option 1: system-only, no user_id)
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,                      -- 'document' | 'youtube' | 'humanml3d'
    external_id TEXT,                               -- youtube video_id (TEXT! sửa bug A2)
    title       TEXT, metadata JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE kb_embeddings (                         -- KB public (tách khỏi memory private)
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,   -- FK thật (sửa A4)
    chunk_index INT NOT NULL DEFAULT 0,
    content     TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_messages_session_seq       ON messages (session_id, seq_id);
CREATE INDEX idx_conversations_user_updated ON conversations (user_id, updated_at DESC);
CREATE INDEX idx_summaries_session          ON summaries (session_id, covers_up_to_seq);
CREATE INDEX idx_summaries_embedding        ON summaries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_user_memory_user           ON user_memory (user_id) WHERE valid = true;
CREATE INDEX idx_kb_emb_document            ON kb_embeddings (document_id);
-- HNSW kb/memory: tạo được lúc bảng rỗng (HNSW không cần training data, khác IVFFlat)
```
**Tenant isolation (sửa A1)** — 2-step, KHÔNG cần user_id trên summaries (D19):
```sql
-- B1: session_ids của user (conversations CÓ user_id)
SELECT session_id FROM conversations WHERE user_id = $1;
-- B2: vector search scoped theo các session đó
SELECT summary_text FROM summaries
WHERE session_id = ANY($ids) AND status='active'
ORDER BY embedding <=> $vec LIMIT 5;
```
session_id từ B1 (lấy theo user đã auth) → B2 chỉ chạm summary của user → không leak. user_id KHÔNG
nằm trong tay LLM (backend bơm $1 từ session) → không có vector tấn công vượt tenant.
**HNSW + filter over-fetch**: cần `iterative_scan` (pgvector 0.8) để recall đúng khi filter `session_id=ANY`.
**Revisit (Phase 7)**: nếu có đường vào DB ngoài tool function (analytics/admin/microservice) → bật
RLS → lúc đó thêm lại user_id (RLS cần cột trên chính bảng). Một-cửa hiện tại chưa cần.

## M.5 — Memory in-session (theo memory-plan)

- **Postgres = nguồn chân lý**; ghi raw NGAY mỗi tin (không đợi disconnect). Cache rebuild được từ DB.
- **Redis `ctx:{session_id}`** chỉ recent raw (KHÔNG chứa summary); write-through; TTL 30-60' idle + LRU.
- **Assemble context** = `[summary chunks (DB)] + [recent raw (cache/DB)] + query`.
- **Summary chunk-based**: trigger theo TOKEN cộng dồn (10k, D13), chạy NỀN, **đóng băng 1 lần**
  (không nén lại → không drift), CAS idempotent (chỉ advance `covers_up_to_seq` nếu mốc mới lớn hơn).
- **Edge A** (summarize fail): retry + cap cứng — recent raw vượt 2× ngưỡng mà chưa nén → fallback,
  không để phình vô hạn.
- **Streaming**: ghi assistant message 1 ROW khi sinh XONG (không ghi từng phần).

## M.6 — Memory xuyên-session (2 tầng)

- **Tầng 1 — `user_memory`** (fact luôn-bật): inject vào system prompt MỌI session. User tự ghi (MVP).
  Conflict → `valid=false` cho fact cũ (so recency). AI-auto trích = phase sau.
  → memory_node load tầng này (rẻ, không vector search), chạy TRƯỚC planner.
- **Tầng 2 — `memory_search` = TOOL của retriever** (không phải việc của memory_node):
  - Giải mâu thuẫn thứ tự: nhu cầu search LTM chỉ biết sau planner → LTM search là 1 **tool**, gọi sau.
  - **2-step scope tenant** (D19, không cần summaries.user_id): B1 lấy `session_id` của user từ
    conversations → B2 `SELECT summary_text FROM summaries WHERE session_id = ANY($ids) AND status='active'
    ORDER BY embedding <=> $vec LIMIT 5`.
  - **Topic vs temporal — tách 2 trục** (bài học cấu trúc-vs-embedding):
    - topic ("project", "squat") → **vector** (`embedding <=>`)
    - thời gian ("hôm bữa", "tuần trước") → planner dịch thành số ngày → **filter cấu trúc**
      (`AND created_at >= now() - $since`), KHÔNG nhồi vào embedding (loãng vector).
    - tool signature: `memory_search(query: str, since: interval | None = None)`. LLM sinh ARGS
      (query + since), DEV viết SQL, backend bơm scope — LLM không ghép SQL.
  - **resume_last_session** (tool riêng cho "tiếp tục session vừa rồi"): B1 lấy session gần nhất
    `WHERE user_id=$1 AND session_id != $current ORDER BY updated_at DESC` (+ `created_at` filter nếu
    planner dịch được mốc thời gian) → lấy CẢ summaries (nếu có) LẪN messages sau mốc summary cuối
    của session đó. Không cần vector nếu chỉ resume theo recency.
  - **kb_search + memory_search gọi SONG SONG** khi cần nhiều nguồn.
  - Kết quả về qua `messages` (ToolMessage) → KHÔNG có field `ltm_results` riêng trong state.

## M.7 — Prompt caching (2 tầng: layout chung + kích hoạt theo provider)

> ⚠️ Provider hiện tại = **DeepSeek** (`llm.py`, ChatOpenAI-compatible). DeepSeek cache
> **TỰ ĐỘNG** theo prefix — KHÔNG cần `cache_control`. Cú pháp `cache_control`/breakpoint là
> của **Anthropic**, chỉ dùng nếu sau này thêm Claude. Đừng viết cache_control cho DeepSeek (vô tác dụng).

### Tầng 1 — Layout context (ÁP DỤNG MỌI PROVIDER, luôn cần)

```
[tools][system + user_memory facts][summary chunks (đóng băng)]   ← TĨNH (prefix cache được)
─────────────────────────────────────────────────────────────────
[recent raw][query]                                               ← ĐỘNG (đổi mỗi lượt)
```
Nguyên tắc (quyết định chi phí, độc lập provider):
- **Tĩnh đầu / động cuối**: phần ít đổi lên trước, phần đổi mỗi lượt xuống cuối.
- **Append-only**: summary chunk mới thêm vào CUỐI phần tĩnh → prefix KHÔNG đổi → cache hit.
  (Đây là lý do summary "đóng băng" — giữ prefix bất biến cho cache.)
- **Prefix bất biến giữa các lần gọi** — provider match prefix theo byte; sửa giữa prefix = mất cache
  (chỉ xảy ra khi GDPR re-summarize — M.8, hiếm, chấp nhận).
- **Chi phí mỗi turn = kích thước recent raw** (phần động, không bao giờ cache) — KHÔNG phải tần
  suất compact. → đây là lý do giữ recent-raw window nhỏ (D13, 10k).

### Tầng 2 — Kích hoạt cache (TÙY PROVIDER)

| Provider | Việc cần làm |
|----------|--------------|
| **DeepSeek** (hiện tại) | KHÔNG làm gì thêm — cache tự động theo prefix. Chỉ cần giữ layout Tầng 1. |
| **Anthropic** (nếu thêm Claude sau) | Đặt `cache_control: ephemeral` sau block tĩnh cuối; block ≥ 1024 token; ≤ 4 breakpoint. |

→ Với DeepSeek, implement = **chỉ làm Tầng 1** (sắp layout + summary đóng băng). Tầng 2 reserve cho
Anthropic. Tầng 1 mới là phần quyết định chi phí và luôn phải làm.

## M.8 — GDPR deletion cascade (CRITICAL, từ đầu — D12)

Xóa raw mà summary đã nén nội dung đó → summary VẪN chứa info đã xóa = vi phạm erasure.
Vì summary đóng băng nên cần luồng deletion riêng:

**Cơ chế mark-dirty = APPLICATION LOGIC (không phải DB cascade).**
`summaries` chỉ FK tới `conversations`, KHÔNG FK tới `messages` → xóa 1 message không tự đụng
summaries. Phải code thủ công:
```
xóa message seq=X
  → UPDATE summaries SET status='dirty'
       WHERE session_id=$1 AND covers_from_seq <= X AND X <= covers_up_to_seq
  → chunk 'dirty' bị LOẠI khỏi context assembly + memory_search NGAY (M.4)
       (trong cửa sổ chờ: dùng raw còn lại — context phình tạm, chấp nhận vì hiếm)
  → re-summarize NỀN từ raw còn lại (bỏ tin đã xóa) → status='active' lại
```

**3 điểm bắt buộc:**
1. **Cửa sổ dirty** (async chưa xong): chunk 'dirty' + embedding của nó bị LOẠI khỏi cả context
   LẪN search. Không có cái này = async lag vẫn rò dữ liệu đã xóa (lỗ hổng critical). → cần cột
   `status` trên summaries (đã thêm M.4).
2. **HARD delete, KHÔNG soft.** `valid=false` của user_memory là cho CONFLICT fact, KHÁC erasure.
   Erasure = xóa thật khỏi DB. **Xóa cả user**: `DELETE conversations WHERE user_id=$1` → FK
   `ON DELETE CASCADE` tự xóa messages + summaries (không cần user_id trên 2 bảng con — D19).
   user_memory cascade qua FK user_id riêng của nó.
3. **Empty-chunk sau xóa** (xóa hết message trong khoảng chunk) → XÓA NGUYÊN chunk + embedding,
   ĐỪNG re-summarize đoạn rỗng.

**2 hệ quả phải biết:**
- Re-summarize = **SỬA giữa prefix → BUST prompt cache** (khác append; hiếm nên chấp nhận).
- **Regen embedding** chunk trong CÙNG transaction với summary_text mới (không thì vector vẫn trỏ
  nội dung đã xóa — note 6 ∩ note 10).

4. Sức khỏe (PT advisor) → encryption at rest + access control + retention (phần lớn là infra Phase 7).

## M.9 — Thứ tự thực hiện

```
1.  Schema mới (Alembic 0002) — M.4, fresh rebuild (không backfill, data hiện tại = test)
2.  Memory layer — M.5 (seq_id, cache, summarize nền, edge A)
3.  Embedding swap → e5-small + prefix — M.4/D10
4.  Intent refactor → PlanOutput 3 trục (required_outputs/resolved_query/needs_retrieval+needs_motion) — M.1
5.  Retriever node — M.2 (input 3 field, tự chọn kb/web/memory song song, empty≠error, no-source refuse)
6.  Tools: kb_search + memory_search (2-step, topic/temporal tách) + web_search — M.2/M.6
7.  Kimodo node riêng (MCP từ edge cứng, planner-gated, song song retriever) — D26
8.  Synthesizer node — M.3b (mode emerge, persona mọi mode, stream B, motion qua tag)
9.  Routing edges (2 cổng độc lập: retriever⟸needs_retrieval, grader⟸required_outputs) — M.2
10. Grader tag-map + TAG_RULES + startup assertion — M.3
11. Clarify động (tool ambiguity metadata → synthesizer hỏi) — M.2b
12. Memory xuyên-session 2 tầng — M.6
13. Prompt caching layout — M.7
14. GDPR deletion cascade — M.8
15. Integration test (PostgreSQL + Redis thật)
```

## M.10 — Còn mở (bàn tiếp trước khi code)

- ~~Danh sách TAG_RULES~~ → ĐÃ CHỐT (M.3): 8 tag PT domain (3 safety + 5 quality). Baseline, mở rộng được.
- ~~Token threshold summarize~~ → **ĐÃ CHỐT: 10k tokens** (recent-raw window = chunk_size, D13).
- ~~`messages.user_id` denormalized?~~ → **ĐẢO NGƯỢC (D19): BỎ user_id khỏi messages + summaries.**
  Sau grill: tenant-scope tập trung ở ít tool function, LLM sinh args không sinh SQL, GDPR qua cascade,
  RLS chưa cần (một-cửa) → user_id thừa. Chỉ giữ session_id. Revisit khi cần RLS (Phase 7).
- ~~Retriever khi nhiều flag true~~ → ĐÃ GIẢI (D16): retriever tự chọn kb/web/memory, song song.
- ~~Planner reconcile (flags vs scope)~~ → ĐÃ CHỐT (D2b, D18): bỏ 3 flags + scope; 1 bit needs_retrieval.

- ~~Synthesizer mode khi bỏ intent~~ → ĐÃ CHỐT (D29): mode emerge từ signals, không enum.
- ~~Motion + retrieval flow~~ → ĐÃ CHỐT (D26 revised): node riêng song song, synthesizer không nhận flag.

**→ 5 NODE ĐÓNG: Memory, Planner, Retriever, Synthesizer, (Grader rule-based — bàn lại sau).**
Không còn điểm mở chặn code. N code được theo §M.9 (15 bước).

**Còn để ngỏ (không chặn):** Grader — Mr. N để bàn lại (rule-based đủ chưa, hay cần nâng). Hiện
giữ rule-based tag-driven (M.3) làm baseline; quyết nâng cấp sau khi chạy thực tế.

## M.11 — Nợ kỹ thuật (defer OK, không block MVP)

- **Grounding check** — grader hiện grade COMPLETENESS (tags) nhưng CHƯA grade "câu trả lời có
  dựa KB không" (chống bịa). Với health advisor đáng có. **Ưu tiên cao hơn query-rewrite** (bịa
  bài tập không có trong KB nguy hiểm hơn retry query kém). Quyết sau.
- **Query-rewrite-on-poor-retrieval** — hiện retry y nguyên query. Rewrite khi retrieval kém = defer.

---

## Phase 6.10 — Pre-deploy Hardening (hiện tại, trước Phase 7)

### 6.10.1 Database: Normalize Messages Table

**Vấn đề hiện tại**

`conversations.messages` là một cột JSONB lưu toàn bộ lịch sử hội thoại dưới dạng JSON array. Mỗi lần có tin nhắn mới, PostgreSQL phải thực hiện `messages || new_messages::jsonb` — server-side append nhưng vẫn là O(n) vì DB phải decode blob cũ, concat, encode lại toàn bộ. Với session 50-100 turns thì TOAST overhead tăng dần theo số lần ghi. Ngoài ra, không thể paginate lịch sử chat mà không pull toàn bộ blob về.

**Giải pháp: Tách bảng `messages`**

```
conversations (session header)
  session_id  UUID PK
  user_id     UUID FK → users(id)
  created_at  TIMESTAMPTZ
  updated_at  TIMESTAMPTZ

messages (one row per message)
  id          UUID PK
  session_id  UUID FK → conversations(session_id)
  role        TEXT  CHECK (role IN ('user', 'assistant'))
  content     TEXT
  metadata    JSONB   -- intent, tokens, v.v.
  created_at  TIMESTAMPTZ

INDEX messages(session_id, created_at)  -- covering index cho pagination
```

Mỗi lần ghi: 1 upsert vào `conversations` (update `updated_at`) + 2 INSERT vào `messages` (user + assistant). Tất cả O(1), không đụng data cũ.

**Cursor-based Pagination**

Load lịch sử chat dùng cursor (timestamp hoặc UUID) thay vì `OFFSET/LIMIT`. Lý do: `OFFSET n` yêu cầu DB scan qua n rows đầu tiên trước khi trả kết quả — chậm dần khi session dài. Cursor-based luôn O(1) vì dùng index:

```sql
-- Load 20 tin nhắn trước cursor (scroll up)
SELECT * FROM messages
WHERE session_id = $1
  AND created_at < $2   -- cursor = created_at của tin nhắn cũ nhất đang hiển thị
ORDER BY created_at DESC
LIMIT 20
```

UI load 20 tin nhắn gần nhất lúc đầu. User scroll lên → gọi tiếp với cursor = `created_at` của tin nhắn cũ nhất đang có → load 20 tin tiếp. Giống pattern của mọi chat app lớn (Slack, Messenger).

**JSONB cho User Profile và Document Metadata**

Hai trường hợp dùng JSONB với lý do khác nhau:

- **User profile**: Các field thay đổi theo thời gian (`age`, `injury_history`, sau này có thể thêm `medication`, `fitness_level`). Dùng cột cố định thì mỗi lần thêm field phải `ALTER TABLE` + migration script. JSONB cho phép thêm field không cần đụng schema. GIN index cho phép query `WHERE profile @> '{"has_injury": true}'` hiệu quả.

- **Document metadata**: Mỗi loại source có metadata khác nhau — research paper có `{doi, author, year}`, video transcript có `{youtube_id, timestamp}`, exercise guide có `{difficulty, equipment}`. Fixed columns sẽ tạo ra hàng loạt `NULL`. JSONB giải quyết heterogeneous schema tự nhiên.

Nguyên tắc chung: field nào **luôn có và query thường xuyên** (`session_id`, `role`, `created_at`) → cột riêng với type chuẩn và B-tree index. Field nào **tùy biến hoặc thay đổi theo thời gian** → JSONB.

**Short-Term Memory (STM)**

Hiện tại STM cứng 3 Q&A pairs trong Redis. **Quyết định (28/05)**: dùng **token budget** thay vì đếm pairs cứng.

Cách hoạt động: Redis vẫn lưu tối đa 3 pairs gần nhất (FIFO write không đổi). Khi đọc, thay vì luôn lấy 3 pairs, hàm `_select_stm_pairs()` lấy từ mới nhất ngược về, cộng dồn token estimate (`len(text) // 4`) cho đến khi đạt budget 1500 tokens. Greeting query ngắn có thể include đủ 3 pairs; clinical query dài với nhiều text thì tự nhiên bị cắt sớm hơn — hành vi đúng về mặt chất lượng context.

Không dùng tokenizer thực (tốn CPU, cần load model). Estimate `// 4` đủ chính xác cho mục đích này — sai số ±20% không ảnh hưởng đến chất lượng response.

Spec code xem Task 2 trong `phases/phase-6.10-predeploy.md`.

**Thứ tự thực hiện**

```
1. Viết migration SQL: tạo messages table + index
2. Viết migration script: backfill data từ conversations.messages JSONB → messages rows
3. Sửa session_store.py: write_session_turn, load_session_messages, list_user_sessions
4. Test: pytest -m integration
5. Sau khi test xanh: DROP COLUMN messages từ conversations
```

Note: codebase dùng raw `asyncpg`, không có SQLAlchemy. Không cần sửa ORM model.

---

### 6.10.2 Pre-deploy Checklist (bắt buộc trước Phase 7)

Các item này đã được document trong `architecture/full-flow-predeploy.md` nhưng chưa implement:

| Item | File cần sửa | Lý do bắt buộc |
|------|-------------|----------------|
| Lock CORS origins | `api/main.py` | Hiện `allow_origins=["*"]` — bất kỳ domain nào cũng gọi được `/chat`. Security risk rõ ràng trước khi expose ra internet. |
| Log file rotation | `shared/logging.py` hoặc config | Không có rotation → log file phình vô hạn trên server. |
| TTS audio cleanup cron | Script mới | VieNeu ghi file audio ra disk, không có cleanup. Server disk đầy theo thời gian. |
| SpeechLLm + SearXNG health checks | `api/health.py` | `/health/detailed` hiện không check 2 services này → ops blind spot khi chúng down. |

---

### 6.10.3 Stop Generation (SSE disconnect detection)

Hiện tại nếu user đóng tab hoặc muốn dừng LLM giữa chừng, graph vẫn chạy hết. Cần thêm `await request.is_disconnected()` check vào vòng lặp `_stream_chat`. Khi client disconnect → break loop → graph bị cancel qua LangGraph cancellation.

---

## Phase 7 — Hybrid Cloud Deployment

> Các mục dưới đây là định hướng kiến trúc. Spec chi tiết và action items sẽ được viết thành từng PHASE-7.x.md riêng khi team bắt đầu Phase 7.

### 7.1 Quyết định giữ SSE, không chuyển WebSocket

**Câu hỏi được đặt ra**: Có nên đổi từ SSE sang WebSocket không, vì sau này có voice command và stop generation?

**Quyết định: Giữ SSE.** Lý do:

**Stop generation** không cần WebSocket. Client đóng `EventSource` → server detect qua `request.is_disconnected()` → cancel graph. Bidirectional channel không cần thiết cho usecase này.

**Voice command**: Câu hỏi then chốt là push-to-talk hay continuous streaming?

- *Push-to-talk* (user nhấn mic, nói, thả ra): Audio được record thành file → POST lên SpeechLLm STT → nhận text → POST `/chat` như bình thường. SSE hoàn toàn đủ.
- *Continuous streaming* (user nói liên tục, server STT real-time): Cần bidirectional → WebSocket phù hợp hơn.

Với PT assistant (healthcare domain), **push-to-talk hợp lý hơn** vì user cần thời gian diễn đạt câu hỏi y tế rõ ràng. Continuous streaming dễ bị noise, phức tạp hơn đáng kể, và không align với usecase thực tế.

**Chi phí của việc đổi sang WebSocket**:
- Load balancer phải support sticky session (hoặc Redis pub/sub fan-out)
- `EventSource` auto-reconnect mất → phải tự viết reconnect logic ở client
- Phase 7 CloudFront: WebSocket qua CDN phức tạp hơn SSE nhiều
- Phải rewrite `api/sse.py`, `ECA_UI/api.js`, và toàn bộ SSE test suite

**Kết luận**: Nếu Phase 7+ có use case conversational voice real-time (user nói, AI ngắt lời, back-and-forth), lúc đó đủ lý do đánh đổi. Có thể build endpoint `/voice` WebSocket riêng mà không đụng `/chat` SSE hiện tại. Không cần quyết định ngay.

---

### 7.2 Phân tách Compute (Service Decomposition)

**Vấn đề**: Các API nhẹ (CRUD sessions) chạy chung container với LangGraph agent. LangGraph container cần GPU-accessible memory, Python deps nặng (langgraph, langchain-openai, asyncpg, v.v.). Chạy một `GET /sessions` đơn giản trên container đó là lãng phí tài nguyên.

**Lưu ý thuật ngữ**: Đây là *service decomposition by compute weight*, không phải CQRS theo nghĩa nghiêm túc. CQRS (Command Query Responsibility Segregation) là pattern tách command model vs query model ở data layer — phức tạp hơn nhiều và không cần thiết ở đây.

**Đề xuất phân tách**:

```
┌─────────────────────────────────────────────────────┐
│  Client (Browser)                                   │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  AWS API Gateway    │  ← CRUD endpoints (serverless)
    │  + Lambda           │    GET /sessions
    │  (pay-per-request)  │    DELETE /sessions
    └─────────────────────┘    POST /sessions/messages (fetch history)

    ┌──────────────────────┐
    │  Application Load    │  ← Heavy endpoints (containerized)
    │  Balancer (ALB)      │    POST /chat (LangGraph agent)
    │  → ECS/EKS container │    GET /health
    └──────────────────────┘
```

**Tại sao CRUD endpoints lên Lambda**:
- Pay-per-request: gần free ở MVP scale
- Auto-scale về 0 khi không có traffic
- Không cần maintain container cho logic đơn giản

**Tại sao `/chat` không qua API Gateway**:
- API Gateway có timeout cứng 29 giây. LangGraph agent chạy 30-60s cho knowledge queries → timeout trước khi xong
- ALB không có timeout giới hạn này, routing trực tiếp đến container

**Prerequisite**: Lambda cần kết nối RDS → phải đặt Lambda trong cùng VPC → cần **RDS Proxy** để tránh connection exhaustion (Lambda stateless = new DB connection mỗi invocation). Không khó nhưng cần setup.

---

### 7.3 TTS: asyncio.create_task → SQS + Worker

**Hiện tại**: `asyncio.create_task` với strong ref (đã fix 26/05). Đủ tốt cho single-server, <100 req/min.

**Vấn đề khi scale**: Khi deploy nhiều ECS task instances, TTS task chạy in-process trên instance nào thì kết quả Redis chỉ có ý nghĩa với request đó. Nếu load balancer route `/tts/{id}/result` poll sang instance khác → 404. Phải dùng sticky session hoặc centralized queue.

**Giải pháp Phase 7: SQS + Worker**

```
Agent (ECS) → SQS queue → TTS Worker (EC2/ECS, pull jobs) → Redis task_result
```

**Tại sao SQS thay vì Kafka hoặc EventBridge**:

*Kafka*: Thiết kế cho throughput cực cao (millions msg/sec), event replay, event sourcing. MVP này TTS queue có thể vài chục request/giờ. Kafka yêu cầu quản lý cluster (hoặc trả tiền Confluent/MSK đắt hơn SQS nhiều lần). Overkill hoàn toàn.

*EventBridge*: Event router/bus — dùng để trigger Lambda khi S3 có file mới, fan-out event sang nhiều service. Không phải task queue — không có visibility timeout, không có Dead Letter Queue tích hợp, không có backpressure. Không phù hợp cho pattern "agent bắn job → worker pick up → retry nếu fail".

*SQS* là đúng tool vì:
- **Visibility timeout**: Worker đang xử lý → message ẩn với worker khác → không bị double-process
- **Dead Letter Queue (DLQ)**: TTS fail 3 lần → tự chuyển sang DLQ → ops có thể debug
- **Managed, pay-per-message**: Không quản lý broker, gần free ở MVP scale
- **Native integration với ECS**: Worker poll SQS bằng boto3, AWS setup sẵn

`celery_app.py` skeleton đã giữ trong codebase từ v2.4.1 chính xác cho usecase này. Phase 7 reactivate, thay broker từ Redis → SQS.

---

### 7.4 Infrastructure as Code — AWS CDK

**Tất cả hạ tầng Phase 7 phải được quản lý qua CDK, không thao tác thủ công trên AWS Console (click-ops).**

Lý do:
- **Reproducibility**: Dev/Staging/Prod đều từ cùng code → không có "works in staging, broken in prod" do config drift
- **Version control**: Infra thay đổi có PR, review, rollback như code
- **`cdk diff`**: Xem chính xác những gì sẽ thay đổi trước khi deploy — tương đương `terraform plan`

**Tại sao CDK thay vì alternatives**:

*Terraform*: Multi-cloud, nhưng cú pháp HCL là một ngôn ngữ riêng cần học. Project đã quyết định AWS-first → CDK native hơn, không cần abstraction layer thêm.

*CloudFormation raw YAML*: CDK compile ra CloudFormation cuối cùng, nhưng viết YAML tay cho VPC + ECS + ALB + SQS + RDS dài, verbose, và error-prone. CDK high-level constructs có security defaults baked in (ví dụ: `ApplicationLoadBalancedFargateService` tự tạo VPC, security group, IAM role đúng minimal-privilege).

**Scope CDK Phase 7**:
```
infra/
  lib/
    vpc-stack.ts         -- VPC, subnets, security groups
    database-stack.ts    -- RDS PostgreSQL + pgvector, RDS Proxy
    cache-stack.ts       -- ElastiCache Redis
    agent-stack.ts       -- ECS Fargate (LangGraph agent container)
    lambda-stack.ts      -- Lambda functions (CRUD endpoints)
    queue-stack.ts       -- SQS queues (TTS jobs, DLQ)
    cdn-stack.ts         -- CloudFront + ALB
  bin/
    app.ts               -- Stack entry point, env params
```

**Tạo `infra/` folder trong repo hiện tại** (không tạo repo riêng) để CDK code và application code versioned cùng nhau.

---

## Tóm tắt Action Items

### Phase 6.10 (làm ngay, trước Phase 7)

| # | Task | File(s) |
|---|------|---------|
| 1 | Migration SQL: tạo `messages` table | `db/migrations/001_normalize_messages.sql` |
| 2 | Migration script: backfill JSONB → rows | `db/migrations/migrate_messages.py` |
| 3 | Sửa session_store.py (4 hàm) | `db/session_store.py` |
| 4 | Cursor-based pagination cho load history | `db/session_store.py`, `api/main.py` |
| 5 | Lock CORS origins | `api/main.py` |
| 6 | Log file rotation | `shared/logging.py` |
| 7 | TTS audio cleanup cron | script mới |
| 8 | SpeechLLm + SearXNG health checks | `api/health.py` |
| 9 | Stop generation: disconnect detection | `api/main.py` |
| 10 | STM token-based sizing (`_select_stm_pairs`) | `nodes/memory.py` |
| 11 | YouTube transcript ingestion → pgvector | `tools/youtube_ingest.py` (mới) |

### Phase 7 (sau 6.10 xong, infra setup riêng)

| # | Task |
|---|------|
| 1 | Init `infra/` CDK project (TypeScript) |
| 2 | VPC + RDS + ElastiCache stacks |
| 3 | ECS Fargate stack cho LangGraph agent |
| 4 | Lambda stack cho CRUD endpoints |
| 5 | SQS + TTS worker (reactivate `celery_app.py` với SQS broker) |
| 6 | CloudFront + ALB stack |
| 7 | DNS + SSL |

---

## Open Questions (cần Owner quyết định trước Phase 7)

1. **Voice feature**: Push-to-talk hay continuous streaming? (Ảnh hưởng đến quyết định SSE vs WebSocket)
2. **LLM provider Phase 7**: Giữ DeepSeek hay thêm Claude/Gemini fallback? (Ảnh hưởng đến latency budget và cost)
3. **Database host**: Supabase managed hay self-hosted RDS? (Supabase dễ setup hơn, RDS linh hoạt hơn về pgvector version)
4. ~~**STM dynamic sizing**: Nếu muốn thay đổi số Q&A pairs trong STM, cần define use case cụ thể trước khi implement~~ → **Đã quyết định**: token budget 1500, xem Task 10 + 6.10.1 STM section
