# ECA Rename Plan — ECA → ECA

> Architect: K | Ngày: 2026-09-01 | Mr. Senryuu duyệt
> Branch đề xuất: `chore/rename-vva-to-eca` (tách từ `feature/langgraph-rewrite`)
> Scope: đổi **mọi tên định danh** ECA → ECA. Không đổi hành vi.
> Tiêu chuẩn thành công: `rg -i '\bvva\b|virtual.verbal' --glob '!node_modules' --glob '!infra/cdk.out' --glob '!.git'` = 0 (trừ `docs/archive/*` + lịch sử git).

---

## 0. Tại sao phải có plan

Repo này đã đổi tên một nửa: folder UI là `ECA_UI`, personas là `eca_default`, tag `Project=ECA`, DB role `eca_user` (status.md:49), nhưng **core vẫn là ECA**:

- 725 match case-insensitive trên 133 file tracked (audit 01/09, chi tiết §1).
- 90 env-var `VVA_*`, 64 SSM `/vva/`, 135 `vva-*` infra, 30 `vva_motion`, 99 file chứa `eca` thường.
- Không rename dứt khoát → người mới đọc `VVA_PG_DSN` + `eca_user` + `Project: ECA` cùng lúc, tưởng 2 dự án khác nhau.

Rename là **breaking change hạ tầng** nếu làm ẩu (SSM, ECR, CFN logical ID, DB user). Plan này tách **đổi tên hiển thị** (an toàn) khỏi **đổi tên hạ tầng** (cần migration), và giữ **backward-compat alias** ít nhất 1 release.

---

## 1. Kiểm kê — thứ gì phải đổi

Audit đầy đủ: `rg --count-matches` 01/09, `git ls-files` (loại `node_modules/`, `infra/cdk.out/`, `*.log`, `.venv*/`).

| # | Nhóm | Pattern cũ | Số match | File tiêu biểu | Đổi thành |
|---|------|-----------|----------|----------------|-----------|
| 1 | **Thư mục gốc + repo** | `ECA` | 12 + remote URL | `docs/architecture/architecture-review.md`, `.mcp.json`, `text-to-motion/DART/mcp_config_example.json` | `ECA` |
| 2 | **Tên dự án hiển thị** | `Embodied Conversational Agent (ECA)` | 10 | `README.md:2`, `.claude/CLAUDE.md:1`, `ECA_UI/frontend/index.html:7`, `docs/index.md` | `Embodied Conversational Agent (ECA)` — chốt tên đầy đủ với Owner trước khi sửa |
| 3 | **Env vars** | `VVA_PG_DSN`, `VVA_PG_DSN_OWNER`, `VVA_PG_DSN_PARAM`, `VVA_PG_STATS`, `VVA_DEV_USERNAME/PASSWORD`, `VVA_PG_STATEMENT_CACHE` (đã deprecated), `__VVA_RETRY_GOOGLE` | 90 | `agenticRAG/.env.example` (11), `agenticRAG/langgraph_agents/db/postgres.py` (10), `agenticRAG/langgraph_agents/alembic/env.py` (7), `scripts/dev_token.ps1` (6) | `ECA_*` (giữ alias `VVA_*` 1 release, xem §3) |
| 4 | **SSM params** | `/vva/neon/dsn`, `/vva/neon/dsn-pooler`, `/vva/llm/*`, `/vva/motion/*` | 64 | `infra/infra/agent_stack.py`, `crud_api_stack.py`, `infra/lambda/layer/shared/db.py`, `agenticRAG/langgraph_agents/shared/env.py` | `/eca/*` |
| 5 | **Docker / compose** | `eca-postgres`, `eca-redis`, `eca-searxng`, `POSTGRES_DB/USER vva`, image `vva-agent`, `vva_motion` | ~60 | `docker-compose.langgraph.yml:4,36,53`, `agenticRAG/Dockerfile:112`, `.github/workflows/deploy-agent.yml:53` | `eca-*` |
| 6 | **Python package** | `text-to-motion/kimodo/vva_motion/` | 30 | `vva_motion/__init__.py`, `agenticRAG/langgraph_agents/nodes/kimodo.py:5`, `agenticRAG/Dockerfile:112` | `eca_motion` |
| 7 | **Infra — Lambda/ECR/CFN** | `vva-agent`, `vva-crud-api`, `vva-characters`, `vva-list-sessions`, `vva-sessions-api`, `vva-streaming-probe`, `vva-db-proxy`, `vva-sg-*`, `VvaAgentStack` etc | ~50 | `infra/infra/*.py`, `infra/app.py:62,65,82,95,120,128,140,160,165` | `eca-*` / `Eca*Stack` |
| 8 | **DB name/user/pass** | `eca` / `eca_dev` / `vva_admin`, `database_name="vva"` | ~40 | `docker-compose.langgraph.yml:6-8`, `config/langgraph.yaml:32`, `infra/infra/database_stack.py:87` | `eca` / `eca_dev` — **riêng Neon đã là `eca_user`** (status.md:49), chỉ còn local + docs |
| 9 | **Frontend / localStorage** | `vva_demo_user`, `vva_session_id`, `vva_avatar_bg`, `vva_*` | ~11 | `ECA_UI/frontend/src/contexts/AvatarBgContext.tsx`, `ChatContext.tsx`, `tests/.../test_a1_a2_a3.py` | `eca_*` |
| 10 | **CI / scripts** | `ECR_REPOSITORY: vva-agent`, `FUNCTION_NAME: vva-agent`, `eca.log` | ~20 | `.github/workflows/deploy-agent.yml`, `scripts/QUICKSTART.md`, `docs/ops/runbook.md` | `eca-*` / `eca.log` |
| 11 | **Docs / comments** | `ECA` trong prose, mermaid, worklogs | 143 `ECA` + 55 `.md` | `docs/tracking/status.md` (15), `docs/worklogs/*.md`, `infra/README.md` (17) | `ECA` |
| 12 | **IAM** | `docs/ops/iam-vva-*.json` | 2 file | — | `iam-eca-*.json` |

**Không đổi:**

- Lịch sử git (`git log --grep=ECA` 27 commit) — không `filter-repo` trừ khi Owner yêu cầu.
- `infra/cdk.out/` (34 file, 3459 match) — regenerate bằng `cdk synth` sau rename.
- `ECA_UI/frontend/node_modules/` (123 file nhiễu hash `AvvaXseg`) — gitignored.
- `*.log` (`eca.log`, `agenticRAG/eca.log`) — gitignored, ephemeral.
- `.venv*/`, `__pycache__/`, `.pytest_cache/`.
- `docs/archive/PHASE-0.md` etc — giữ nguyên như archive **hoặc** rename tùy Owner; plan mặc định **giữ archive** để lịch sử đọc được.

**Mâu thuẫn đã tồn tại (ghi chú để không đổi nhầm):**

- `infra/app.py:52` `Tags.of(app).add("Project", "ECA")` **đã là ECA** — không sửa.
- `ECA_UI/` **đã là ECA** — không đổi.
- `personas/eca_default|friendly|clinical` **đã là eca_** — không đổi.
- `eca_user` (Neon role) **đã là eca_** — local `eca` là cái còn lại cần đổi.

---

## 2. Quy ước đặt tên mới — chốt trước khi code

| Khái niệm | Cũ | Mới | Ghi chú |
|-----------|----|----|---------|
| Tên đầy đủ | Embodied Conversational Agent | **Embodied Conversational Agent** — Owner chốt | Nếu Owner muốn giữ "Embodied Conversational Agent" nhưng viết tắt ECA thì ghi rõ ở README |
| Viết tắt | ECA | **ECA** (uppercase) | Trong code `ECA_*`, trong prose `ECA` |
| Repo | `ECA` | **`ECA`** (Owner chọn) — hoặc `Embodied-Conversational-Agent` nếu muốn tên đầy đủ | Đổi trên GitHub Settings → thông báo team `git remote set-url` |
| Folder local | `ECA` | `ECA` | Sau khi đổi repo, `git clone` mới ra `ECA`, folder cũ rename thủ công |
| Env vars | `VVA_*` | `ECA_*` | Alias `VVA_*` giữ 1 release (§3) |
| SSM prefix | `/vva/` | `/eca/` | Copy param → dual-read → xóa cũ |
| Docker | `vva-*` | `eca-*` | Compose `container_name`, `database_name` |
| ECR / Lambda | `vva-agent` | `eca-agent` | Repo ECR mới, không rename được |
| CFN stacks | `Vva*Stack` | `Eca*Stack` | Đổi construct ID = đổi logical ID (§3 cảnh báo) |
| Python pkg | `vva_motion` | `eca_motion` | `import eca_motion` |
| localStorage | `vva_*` | `eca_*` | Migration đọc cũ ghi mới |
| Log file | `eca.log` | `eca.log` | `.env` `LOG_FILE` |
| IAM file | `iam-vva-*.json` | `iam-eca-*.json` | Rename + cập nhật doc tham chiếu |

**Case mapping chi tiết:**

- `ECA` → `ECA`
- `eca` → `eca`
- `Vva` (construct ID) → `Eca` (ví dụ `VvaAgentStack` → `EcaAgentStack`)
- `ECA` → `ECA`
- `Embodied Conversational Agent` → `Embodied Conversational Agent`

---

## 3. Rủi ro & quyết định kiến trúc

### R1 — Env var rename làm rơi về local DB (nguy hiểm nhất)

`db/postgres.py:_resolve_dsn` fallback silent về `postgresql://vva:eca_dev@localhost:5433/vva` khi thiếu `VVA_PG_DSN`. Đổi tên đột ngột → mọi dev chưa update `.env` lặng lẽ ghi nhầm DB. Đã từng xảy ra khi xóa `agentic_rag_gemini/.env` (CLAUDE.md ghi rõ).

**Quyết định:** dual-read ít nhất 1 release.

```python
# db/postgres.py, shared/env.py, alembic/env.py
# Đọc ECA_* trước, fallback VVA_* kèm deprecation warning, cuối cùng mới fallback local.
dsn = os.getenv("ECA_PG_DSN") or os.getenv("VVA_PG_DSN")
if os.getenv("VVA_PG_DSN") and not os.getenv("ECA_PG_DSN"):
    logger.warning("VVA_PG_DSN is deprecated — rename to ECA_PG_DSN")
```

Áp dụng cho `ECA_PG_DSN`, `ECA_PG_DSN_OWNER`, `ECA_PG_DSN_PARAM`, `ECA_PG_STATS`. Tương tự `ECA_DEV_USERNAME/PASSWORD` trong `scripts/dev_token.ps1`.

Xóa alias chỉ khi `rg -i VVA_ --glob '!archive'` = 0 và đã thông báo team 2 tuần.

### R2 — SSM `/vva/` → `/eca/` (không rename được)

SSM param name immutable. Phải tạo mới + cấp quyền.

**Quyết định:** copy → dual-read → xóa.

1. `aws ssm put-parameter --name /eca/neon/dsn --type SecureString --value "$(aws ssm get-parameter --name /vva/neon/dsn --with-decryption --query Parameter.Value --output text)"` (lặp cho 6 param ở infra/README.md bảng).
2. Code đọc `ECA_*_PARAM` trước, fallback `/vva/` (ví dụ `llm.py`, `db.py` layer, `agent_stack.py`).
3. Update IAM policy `ssm:GetParameter` cho `/eca/*` (giữ `/vva/*` tới khi xóa).
4. Sau deploy verify `rg /vva/ infra/` = 0, mới `aws ssm delete-parameter --name /vva/...`.

### R3 — ECR `vva-agent` → `eca-agent` (không rename được)

**Quyết định:** `cdk deploy EcaAgentStack -c agent_bootstrap=1` tạo repo mới `eca-agent`, CI push sang repo mới, mới switch Lambda.

1. Infra `agent_stack.py`: `repository_name="eca-agent"` (giữ alias đọc `vva-agent` trong 1 deploy nếu cần rollback).
2. `.github/workflows/deploy-agent.yml`: `ECR_REPOSITORY: eca-agent`, `FUNCTION_NAME: eca-agent`.
3. CI phải có quyền `ecr:CreateRepository` hoặc tạo thủ công trước.

### R4 — CFN stack/construct rename đổi logical ID → replacement

`VvaAgentStack` → `EcaAgentStack` đổi logical ID của mọi resource trong stack. `cdk diff` sẽ hiện **replacement** (xóa + tạo mới) cho DynamoDB `vva-motion-jobs`, IAM roles, v.v. — mất dữ liệu nếu deploy vô tội vạ.

**Quyết định:** 2 lựa chọn, Owner chốt:

- **A (an toàn): giữ StackName CFN cũ, chỉ đổi construct ID hiển thị.** Trong `app.py` truyền `stack_name="VvaAgentStack"` khi construct là `EcaAgentStack`. Cần `overrideLogicalId` cho resource stateful. Phức tạp, dễ sót.
- **B (sạch, khuyến nghị cho dự án chưa production): tạo stack mới `Eca*`, deploy song song, cutover DNS/API Gateway, rồi `cdk destroy Vva*`.** Tốn thời gian nhưng không risk replacement. Phù hợp vì Track 2 chưa production, Track 1 chưa deploy bao giờ.

Plan mặc định chọn **B**. Nếu Owner chọn A, phải chạy `cdk diff` + review từng replacement trước deploy.

### R5 — DB local `eca` → `eca`

Neon prod đã là `eca_user`/`neondb` (không đổi). Chỉ còn `docker-compose.langgraph.yml` (`POSTGRES_DB: vva`, `POSTGRES_USER: vva`, `pgdata` volume) và `config/langgraph.yaml`, `alembic.ini`.

**Quyết định:** đổi compose + config, **không migrate volume prod**. Local dev chạy `docker compose down -v && up -d` là xong (data local). Ghi rõ trong `scripts/QUICKSTART.md`.

### R6 — `vva_motion` → `eca_motion`

Đổi tên folder + mọi import + Dockerfile COPY.

**Quyết định:** `git mv text-to-motion/kimodo/vva_motion text-to-motion/kimodo/eca_motion` + `rg vva_motion` sweep. Giữ shim 1 release nếu muốn:

```python
# eca_motion/__init__.py
import sys
sys.modules["vva_motion"] = sys.modules[__name__]  # compat
```

Nhưng vì chưa có external consumer, có thể bỏ shim.

### R7 — Frontend localStorage `vva_*`

**Quyết định:** đọc cả 2, ghi mới.

```ts
const get = (k: string) => localStorage.getItem(`eca_${k}`) ?? localStorage.getItem(`vva_${k}`);
const set = (k: string, v: string) => localStorage.setItem(`eca_${k}`, v);
```

Xóa `vva_*` sau 1 release hoặc khi user logout.

---

## 4. Thứ tự theo mức độ an toàn — làm từ an toàn nhất trước

> Cùng 725 điểm chạm như §1, nhưng sắp lại theo **blast radius + khả năng rollback**.
> Tier càng thấp càng làm trước — mỗi tier xong đều có thể dừng, demo, và revert rẻ.
> Quy tắc: **chỉ lên tier sau khi tier trước đã xanh test gate §6.2**.

| Tier | Mức an toàn | Nhóm (§1) | Blast radius nếu sai | Rollback | Làm khi nào |
|------|-------------|-----------|----------------------|----------|-------------|
| **S0** | 🟢 **Zero-risk** — chỉ chữ, không chạy | #11 Docs/prose/comments (143 `ECA` + 55 `.md`), #1 path hard-code trong doc archive | Không ai chạy code này | `git revert` 1 commit | Ngay, không cần review infra |
| **S1** | 🟢 **An toàn** — hiển thị, không logic | #2 Tên dự án hiển thị (`README:2`, `index.html:7`, `amplify/backend.ts: restApiName`), #10 `eca.log`→`eca.log`, #12 `iam-vva-*.json` rename, `start_services.py` title, `health-test` | User thấy chữ mới, không hỏng API | `git revert` | Cùng S0 |
| **S2** | 🟡 **An toàn có điều kiện** — local-only | #5 Docker compose `eca-postgres/redis/searxng` + `POSTGRES_DB/USER`, #8 DB local `eca`→`eca`, `config/langgraph.yaml`, `alembic.ini`, `requirements-*.txt` comment | Chỉ dev local; prod Neon đã là `eca_user` nên không chạm | `docker compose down -v && git revert` — mất data local nhưng không mất prod | Sau S1, 0.5 ngày |
| **S3** | 🟡 **Trung bình** — code, có shim/alias | #6 `vva_motion`→`eca_motion` (30 file) với shim `sys.modules["vva_motion"]`, #9 `localStorage vva_*`→`eca_*` (đọc cả 2, ghi mới), `ECA_UI/frontend/src/lib/api.ts` không đổi | Import lỗi → test đỏ ngay, không lọt prod nếu CI xanh | `git mv eca_motion vva_motion` + revert | Sau S2, 0.5 ngày |
| **S4** | 🟠 **Nhạy cảm** — config, cần dual-read | #3 Env vars `VVA_*`→`ECA_*` (90 match), #3 `VVA_PG_DSN_OWNER/PARAM/STATS`, `VVA_DEV_USERNAME` (dual-read + warning trong `postgres.py`, `env.py`, `preflight.py`, `dev_token.ps1`) | **Cao nhất nếu làm ẩu**: thiếu dual-read → fallback silent về `localhost:5433/vva` → ghi nhầm DB (đã từng xảy ra, §R1) | Revert 1 commit, vì alias cũ vẫn đọc được | Sau S3, 1 ngày — **bắt buộc review K** |
| **S5** | 🔴 **Rủi ro hạ tầng** — không rename được, phải tạo mới | #4 SSM `/vva/`→`/eca/` (64), #7 ECR `vva-agent`→`eca-agent`, Lambda `vva-*`→`eca-*`, `Vva*Stack`→`Eca*Stack` (§R2–R4) | Tạo resource mới, sai thì tốn tiền + 403 CloudFront + Lambda không kéo image | SSM/ECR cũ **giữ lại** tới Phase 5 nên revert được; CFN nếu chọn B (stack mới) thì `cdk destroy Eca*` là xong, nếu chọn A thì replacement mất DynamoDB | Sau S4, cần **AWS creds** (đang chặn cứng — status.md), Owner duyệt `cdk diff` |
| **S6** | 🔴 **Không thể rollback** — xóa alias | Xóa fallback `VVA_*` trong code, `aws ssm delete-parameter /vva/*`, xóa ECR `vva-agent`, `cdk destroy Vva*Stack` | Vĩnh viễn, mọi `.env` cũ hỏng | Không rollback — phải tạo lại thủ công | **Sau 2–4 tuần** kể từ S4+S5, khi `rg -i VVA_ --glob '!archive'` = 0 và team confirm |

**Thứ tự khuyến nghị (đè lên phase cũ):**

```
S0+S1 (0.5 ngày, PR1) → S2+S3 (0.5 ngày, cùng PR1) → S4 (1 ngày, PR1) → S5 (1 ngày, PR2, cần creds) → S6 (0.2 ngày, sau 2–4 tuần)
```

- **Làm ngay không sợ (S0–S1):** `README.md`, `.claude/CLAUDE.md`, `docs/**/*.md` (trừ `docs/archive/`), `ECA_UI/frontend/index.html`, `amplify/backend.ts`, `iam-*.json`, `eca.log`→`eca.log`. 0 dòng logic đổi.
- **Làm local an toàn (S2–S3):** `docker-compose`, `vva_motion` (có shim), `localStorage`. Hỏng thì chỉ N tự fix, không ảnh hưởng prod.
- **Cửa ngõ nguy hiểm (S4):** env vars — **không được** đổi 1 phát `s/ECA/ECA/g`. Phải dual-read + warning, test với cả `.env` cũ và mới.
- **Hạ tầng (S5):** tách PR riêng, bắt buộc `cdk diff` + Owner duyệt. Không chạy `cdk deploy --all`.
- **Dọn dẹp (S6):** chỉ khi S4+S5 đã sống 1 release không warning.

> Nếu nguồn lực hạn chế: chỉ làm **S0→S4** là đã đạt 80% giá trị (người mới không còn thấy 2 tên lẫn lộn), S5+S6 để khi có AWS creds.

---

## 4b. Kế hoạch thực thi — 5 phase (giữ theo dependency cũ để tham chiếu)

> §4 là thứ tự **an toàn** để làm. §4b là thứ tự **dependency kỹ thuật** cũ — giữ lại để N tra file.

### Phase 0 — Chuẩn bị (0.5 ngày, không code) — thuộc S0

- [ ] Owner chốt tên đầy đủ (ECA = gì) + tên repo mới (`ECA` hay `Embodied-Conversational-Agent`).
- [ ] Tạo branch `chore/rename-vva-to-eca` từ `feature/langgraph-rewrite`.
- [ ] Chạy audit script tạo baseline: `rg -i vva --count-matches | tee /tmp/vva-baseline.txt` (725 match kỳ vọng).
- [ ] Ghi `docs/worklogs/DD-MM-YYYY.md` entry "rename plan approved".
- [ ] Thông báo team: sắp rename, chuẩn bị đổi `.env` và `git remote`.

**Gate:** Owner duyệt bảng §2.

### Phase 1 — Code & config (dual-read) — 1 ngày — thuộc S4

**Mục tiêu:** mọi `ECA_*` hoạt động, `VVA_*` vẫn chạy (warning).

| File | Việc |
|------|------|
| `agenticRAG/.env.example` | Thêm `ECA_*` trước, giữ `VVA_*` comment "deprecated — will be removed 2026-10-01" |
| `agenticRAG/langgraph_agents/db/postgres.py` | `_resolve_dsn`: `ECA_PG_DSN` → fallback `VVA_PG_DSN` + warning |
| `agenticRAG/langgraph_agents/alembic/env.py` | Tương tự cho `ECA_PG_DSN_OWNER`, `ECA_PG_DSN` |
| `agenticRAG/langgraph_agents/shared/env.py` | Update docstring + `require()` hỗ trợ alias |
| `agenticRAG/langgraph_agents/shared/preflight.py` | Đọc `ECA_PG_DSN_PARAM` fallback `VVA_PG_DSN_PARAM` |
| `infra/infra/agent_stack.py` | Đọc SSM `/eca/...` trước, fallback `/vva/...`; env `ECA_PG_DSN_PARAM` |
| `infra/infra/crud_api_stack.py` | Tương tự |
| `infra/lambda/layer/shared/db.py` | `DB_MODE` param `/eca/neon/dsn*` fallback `/vva/...` |
| `scripts/dev_token.ps1` | `ECA_DEV_USERNAME/PASSWORD` fallback `VVA_*` |
| `scripts/sync_personas_to_db.py`, `upload_characters_to_s3.py` | Đổi env var đọc |
| `agenticRAG/langgraph_agents/api/main.py` | `ECA_PG_STATS` |
| `ECA_UI/frontend/src/lib/api.ts` | Không đổi (đã dùng `VITE_API_BASE_URL`, không phải ECA) |
| `ECA_UI/frontend/src/contexts/*` | `eca_*` với fallback đọc `vva_*` (§R7) |
| `.github/workflows/deploy-agent.yml` | Thêm `ECA` env, giữ `ECA` alias nếu runner cũ |
| `tests/**` | Update fixtures `conftest.py`, `test_rls_policies.py` (6 match `VVA_PG_DSN`) để test cả 2 |

**Script hỗ trợ:**

```bash
# scripts/rename_vva_to_eca.py — dry-run sweep
# 1. git ls-files | xargs rg -l -i '\bvva\b'
# 2. áp mapping §2, bỏ qua docs/archive/*, infra/cdk.out, node_modules
# 3. in diff, không ghi file khi --dry-run
python scripts/rename_vva_to_eca.py --dry-run | tee /tmp/rename.diff
```

**Verify:**

- `python -m pytest tests/langgraph_agents -m unit -q` — 331 passed kỳ vọng (firstconda).
- `rg -n "VVA_PG_DSN" agenticRAG/langgraph_agents/db/postgres.py` còn nhưng có comment deprecated.
- Chạy backend với `.env` cũ (ECA) → log warning nhưng chat vẫn chạy + `/health/detailed` ok.

### Phase 2 — Python package & Docker — 0.5 ngày — thuộc S2+S3

| File | Việc |
|------|------|
| `text-to-motion/kimodo/vva_motion/` | `git mv` → `eca_motion/` |
| `agenticRAG/langgraph_agents/nodes/kimodo.py:5` | `from eca_motion` (hoặc `import eca_motion`) |
| `agenticRAG/Dockerfile:112` | `COPY text-to-motion/kimodo/eca_motion ./eca_motion` |
| `text-to-motion/kimodo/Dockerfile*` | `COPY eca_motion` |
| `text-to-motion/kimodo/worker.py` | import path |
| `tests/langgraph_agents/test_kimodo*.py` | import + mock path |
| `docker-compose.langgraph.yml` | `container_name: eca-postgres/eca-redis/eca-searxng`, `POSTGRES_DB: eca`, `POSTGRES_USER: eca`, `POSTGRES_PASSWORD: eca_dev`, `INSTANCE_NAME: eca-search` |
| `config/langgraph.yaml:32` | `postgresql://eca:eca_dev@...` |
| `agenticRAG/langgraph_agents/alembic.ini:7` | `postgresql+asyncpg://eca:eca_dev@...` |
| `requirements-langgraph.txt`, `agenticRAG/requirements-agent-runtime.txt` | comment `VVA_PG_DSN_PARAM` → `ECA_*` |
| `README.md`, `docs/ops/runbook.md`, `scripts/QUICKSTART.md` | `docker exec eca-postgres ...` |

**Verify:**

- `docker compose -f docker-compose.langgraph.yml config` — không còn `vva-`.
- `python -c "import eca_motion; print(eca_motion.__file__)"` ok.
- `rg vva_motion --glob '!cdk.out' --glob '!.git'` = 0.

### Phase 3 — Infra (SSM, Lambda, ECR, CFN) — 1 ngày, cần AWS creds — thuộc S5

> ⚠️ Phase này **không chạy** nếu chưa có AWS creds (status.md: chặn cứng). Code có thể merge, deploy để sau.

1. **SSM:** tạo `/eca/*` song song `/vva/*` (6 param). Script:

   ```bash
   for p in neon/dsn neon/dsn-pooler llm/deepseek-api-key llm/gemini-api-keys motion/signing-key-pem motion/hash-secret; do
     aws ssm get-parameter --name /vva/$p --with-decryption --query Parameter.Value --output text > /tmp/val
     aws ssm put-parameter --name /eca/$p --type SecureString --value file:///tmp/val --overwrite
   done
   ```

2. **CDK stacks:** theo lựa chọn §R4.

   - Nếu **B (khuyến nghị):** tạo `EcaAgentStack`, `EcaCrudApiStack`, `EcaCharacterStack`, `EcaAssetStack`, `EcaVpcStack` (rename file `infra/infra/*_stack.py` class `Vva*` → `Eca*`). Giữ `Vva*` tới khi cutover xong.
   - Nếu **A:** giữ `stack_name`, chỉ đổi construct ID + `overrideLogicalId` cho DynamoDB/IAM.

3. **ECR/Lambda:** `agent_stack.py: repository_name="eca-agent"`, `function_name="eca-agent"`; `crud_api_stack.py: function_name="eca-crud-api"`; `character_stack.py: "eca-characters"`; `lambda_stack.py: "eca-list-sessions"` etc; `database_stack.py: "eca-db-proxy"`.

4. **IAM:** `docs/ops/iam-eca-*.json` + policy `ssm:GetParameter` cho `/eca/*`.

5. **CI:** `.github/workflows/deploy-agent.yml` → `ECR_REPOSITORY: eca-agent`.

**Verify:**

- `cdk synth --no-lookups 2>&1 | rg -i vva` — chỉ còn comment deprecated.
- `cdk diff EcaAgentStack` — review replacement, không có unexpected deletion.
- `aws ssm get-parameter --name /eca/neon/dsn-pooler --with-decryption` ok.

### Phase 4 — Repo, docs, CI, cleanup hiển thị — 0.5 ngày — thuộc S0+S1

| File | Việc |
|------|------|
| `README.md:2` | `Embodied Conversational Agent (ECA)` → `Embodied Conversational Agent (ECA)` |
| `.claude/CLAUDE.md:1,69,123` | Đổi title + `VVA_PG_DSN` → `ECA_PG_DSN` + `aws-*` skill description |
| `docs/tracking/status.md` | `ECA — Status` → `ECA — Status`, `docker eca-postgres` → `eca-postgres`, `VVA_PG_STATS` → `ECA_PG_STATS`, `eca.log` → `eca.log` |
| `docs/ops/*.md`, `docs/worklogs/*.md`, `docs/architecture/*.md`, `docs/phases/*.md`, `docs/plans/*.md` | Sweep `ECA` → `ECA` (giữ archive nếu Owner chọn) |
| `infra/README.md` | Toàn bộ SSM `/vva/` → `/eca/`, stack names, deploy commands |
| `ECA_UI/frontend/index.html:7` | meta description |
| `ECA_UI/frontend/amplify/backend.ts` | `restApiName: 'ECA Auth API'` |
| `ECA_UI/frontend/src/components/SettingsContent.tsx` | `{ id: 'eca', name: 'ECA' }` |
| `ECA_UI/test-ui/health-test/*` | `ECA` string |
| `start_services.py:2` | `title = f"ECA · {name}"` |
| `scripts/QUICKSTART.md` | `docker exec eca-postgres` |
| `infra/app.py` | `VvaVpcStack` → `EcaVpcStack` etc + comment header |
| `docs/ops/iam-vva-*.json` | Rename file + nội dung |
| `text-to-motion/DART/mcp_config_example.json` | `D:/.../ECA/...` |

**Repo rename (Owner làm):**

```bash
# Trên GitHub: Settings → General → Repository name → ECA → Rename
# GitHub tự redirect ECA → ECA (giữ 1 thời gian), nhưng vẫn nên:
git remote set-url origin https://github.com/Beginner-Mon/ECA.git
# Thông báo team:
git remote -v  # verify
```

**Folder local (mỗi dev):**

```bash
# Đóng IDE, rồi:
Move-Item "D:\Swin documents\ECA" "D:\Swin documents\ECA"
# Mở lại IDE ở path mới, kiểm tra .env còn không (gitignored, phải copy thủ công nếu move lỗi)
```

**Verify:**

- `rg -i '\bvva\b' --glob '!docs/archive' --glob '!infra/cdk.out' --glob '!.git' --glob '!node_modules'` = 0.
- `npm --prefix ECA_UI/frontend run build` + `tsc --noEmit` 0 lỗi.
- `python -m pytest tests/ -m unit -q` 331 passed.

### Phase 5 — Xóa alias & đóng (sau 2–4 tuần) — thuộc S6

- [ ] Confirm không còn `.env` nào dùng `VVA_*` (hỏi team, check CI secrets).
- [ ] Xóa fallback `VVA_*` trong `db/postgres.py`, `alembic/env.py`, `shared/env.py`, `dev_token.ps1`, frontend `localStorage`.
- [ ] `aws ssm delete-parameter --name /vva/...` (6 param).
- [ ] Xóa ECR `vva-agent` (sau khi confirm không rollback).
- [ ] `cdk destroy Vva*Stack` nếu chọn B.
- [ ] Update `docs/tracking/tech-debt.md` — xóa mục rename.
- [ ] Ghi worklog `docs/worklogs/DD-MM-YYYY.md`: "ECA→ECA alias removed".

---

## 5. Danh sách file chi tiết (để N tick)

> Sinh từ `git ls-files | xargs rg -l -i vva` 01/09. Đánh ✓ khi xong.

**Env / config (12):**
- [ ] `agenticRAG/.env.example` (11 ECA)
- [ ] `agenticRAG/langgraph_agents/alembic.ini` (1)
- [ ] `agenticRAG/langgraph_agents/alembic/env.py` (7)
- [ ] `agenticRAG/langgraph_agents/shared/env.py` (1)
- [ ] `agenticRAG/langgraph_agents/shared/preflight.py` (2)
- [ ] `agenticRAG/langgraph_agents/db/postgres.py` (10)
- [ ] `agenticRAG/Dockerfile` (1 ECA + 5 vva)
- [ ] `docker-compose.langgraph.yml` (9)
- [ ] `config/langgraph.yaml` (3)
- [ ] `requirements-langgraph.txt` (1)
- [ ] `agenticRAG/requirements-agent-runtime.txt` (1)
- [ ] `ECA_UI/frontend/.env.example` (1)

**Infra (12):**
- [ ] `infra/app.py` (1 + 7 stack refs)
- [ ] `infra/infra/agent_stack.py` (1 + 13)
- [ ] `infra/infra/api_gateway_stack.py` (3)
- [ ] `infra/infra/asset_stack.py` (1)
- [ ] `infra/infra/character_stack.py` (1)
- [ ] `infra/infra/crud_api_stack.py` (1 + 14)
- [ ] `infra/infra/database_stack.py` (1 + 11)
- [ ] `infra/infra/lambda_stack.py` (2 + 12)
- [ ] `infra/infra/rest_api_stack.py` (2)
- [ ] `infra/infra/kimodo_ecs_stack.py` (vva_motion ref)
- [ ] `infra/infra/vpc_stack.py` (tag check)
- [ ] `infra/README.md` (3 + 17)

**Frontend (8):**
- [ ] `ECA_UI/frontend/amplify/backend.ts`
- [ ] `ECA_UI/frontend/src/components/SettingsContent.tsx`
- [ ] `ECA_UI/frontend/src/contexts/AvatarBgContext.tsx`
- [ ] `ECA_UI/frontend/src/contexts/ChatContext.tsx`
- [ ] `ECA_UI/frontend/src/contexts/GraphicsContext.tsx`
- [ ] `ECA_UI/frontend/src/lib/api.ts`
- [ ] `ECA_UI/frontend/vite.config.ts`
- [ ] `ECA_UI/frontend/index.html`

**Backend (15):**
- [ ] `agenticRAG/langgraph_agents/api/main.py` (3)
- [ ] `agenticRAG/langgraph_agents/api/billing_local.py` (2)
- [ ] `agenticRAG/langgraph_agents/state.py` (1)
- [ ] `agenticRAG/langgraph_agents/nodes/kimodo.py`
- [ ] `agenticRAG/langgraph_agents/db/session_store.py`
- [ ] `agenticRAG/langgraph_agents/mcp/client.py`
- [ ] `start_services.py` (2)
- [ ] `scripts/dev_token.ps1` (6)
- [ ] `scripts/sync_personas_to_db.py` (1)
- [ ] `scripts/upload_characters_to_s3.py` (1)
- [ ] `ECA_UI/test-ui/health-test/app.js` + `index.html`

**Tests (10):**
- [ ] `tests/langgraph_agents/conftest.py`
- [ ] `tests/langgraph_agents/test_rls_policies.py` (6)
- [ ] `tests/langgraph_agents/test_crud_pooled_integration.py`
- [ ] `tests/langgraph_agents/test_kimodo_node.py`
- [ ] `tests/langgraph_agents/test_a1_a2_a3.py`
- [ ] `tests/infra/test_motion_route_infra.py`
- [ ] `tests/infra/test_kimodo_sg_replaceable.py`
- [ ] `tests/infra/test_dockerignore_allows_copied_paths.py`
- [ ] `tests/text-to-motion/test_worker_loop.py`
- [ ] `tests/conftest.py`

**Docs (30+):**
- [ ] `README.md` (2)
- [ ] `.claude/CLAUDE.md` (3)
- [ ] `docs/tracking/status.md` (5 + 15 vva)
- [ ] `docs/ops/runbook.md` (4 + 26)
- [ ] `docs/ops/handover-19-08-2026.md` (6)
- [ ] `docs/ops/neon-migration-us-east-1.md` (2)
- [ ] `infra/lambda/layer/shared/db.py` (1)
- [ ] ... sweep toàn bộ `docs/**/*.md` (55 file)

**Package:**
- [ ] `text-to-motion/kimodo/vva_motion/` → `eca_motion/` (git mv)
- [ ] `text-to-motion/kimodo/Dockerfile` + `Dockerfile.prod`

**CI:**
- [ ] `.github/workflows/deploy-agent.yml`
- [ ] `.github/workflows/release-tests.yml` (check vva-agent ref)

---

## 6. Script & kiểm thử

### 6.1 Sweep script (N chạy local)

```bash
# 1. Dry-run — xem diff trước khi ghi
python scripts/rename_vva_to_eca.py --dry-run --exclude docs/archive --exclude infra/cdk.out

# 2. Áp dụng
python scripts/rename_vva_to_eca.py --apply

# 3. Kiểm tra còn sót
rg -i '\bvva\b' --glob '!docs/archive/**' --glob '!infra/cdk.out/**' --glob '!.git/**' --glob '!**/node_modules/**' --hidden
# kỳ vọng: 0

rg -n 'Virtual.Verbal' --glob '!docs/archive/**'
# kỳ vọng: 0 (trừ worklog lịch sử nếu giữ)

# 4. Test
python -m pytest tests/langgraph_agents -m unit -q          # 331 passed
python -m pytest tests/infra -q                              # infra synth tests
npm --prefix ECA_UI/frontend run build                       # tsc + vite
cdk synth --no-lookups 2>&1 | rg -i vva                      # chỉ warning deprecated

# 5. Docker
docker compose -f docker-compose.langgraph.yml config | rg vva  # 0
```

Nội dung `scripts/rename_vva_to_eca.py` (30 dòng, stdlib):

- Đọc `git ls-files` (chỉ tracked).
- Mapping: `VVA_`→`ECA_`, `vva-`→`eca-`, `vva_`→`eca_`, `/vva/`→`/eca/`, `eca.log`→`eca.log`, `ECA`→`ECA`, `eca`→`eca`, `Vva`→`Eca`, `ECA`→`ECA`, `Embodied Conversational Agent`→`Embodied Conversational Agent`.
- Skip: `docs/archive/**`, `infra/cdk.out/**`, `*.log`.
- `--dry-run` in diff unified, `--apply` ghi file.

### 6.2 Test gate

| Gate | Lệnh | Pass khi |
|------|------|----------|
| Unit | `pytest -m unit -q` | 331 passed, 0 failed (firstconda) |
| Infra synth | `cdk synth --no-lookups` | 0 error, `rg -i vva` chỉ còn comment deprecated |
| Frontend | `npm run build` | `tsc -b` 0 lỗi, `vite build` ok |
| Docker | `docker compose config` | không còn `vva-` |
| E2E smoke | `curl localhost:8000/health/detailed` | `status: ok` hoặc `degraded` (TTS optional) |
| Chat smoke | `curl -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -d '{"query":"xin chào"}'` | SSE `stage`→`token`→`done` |

---

## 7. Rollback

| Phase | Rollback |
|-------|----------|
| 1 (code dual-read) | Revert commit, không ảnh hưởng infra (dual-read nên cũ/mới đều chạy) |
| 2 (package/docker) | `git mv eca_motion vva_motion`, revert compose; local `docker compose down -v` |
| 3 (SSM/ECR) | SSM `/vva/` vẫn giữ nên revert code là chạy lại; ECR `vva-agent` chưa xóa nên rollback tag |
| 4 (repo) | GitHub Settings → rename lại `ECA`; `git remote set-url origin .../ECA.git`; folder local rename ngược |
| 5 (xóa alias) | Không rollback được SSM/ECR đã xóa — phải tạo lại. Vì vậy Phase 5 chỉ làm sau 2–4 tuần verify |

---

## 8. Lịch & phân công

| Phase | Thời gian | Ai | Ghi chú |
|-------|-----------|----|---------|
| 0 duyệt | 0.5 ngày | K + Owner | Chốt tên đầy đủ + repo name |
| 1 code dual-read | 1 ngày | N | Không cần AWS creds, làm ngay |
| 2 package/docker | 0.5 ngày | N | Cùng PR với Phase 1 |
| 3 infra | 1 ngày | N + K review | Cần AWS creds, có thể tách PR riêng |
| 4 repo/docs | 0.5 ngày | N (code) + Owner (repo rename) | Sau khi Phase 1 merge |
| 5 xóa alias | 0.2 ngày | N | Sau 2–4 tuần |

**Tổng:** ~3.5 ngày code + 2–4 tuần chờ alias.

**PR đề xuất:**

- PR1: Phase 1+2 (code + package + docker + tests) — `chore/rename-vva-to-eca-code` — review thường.
- PR2: Phase 3 (infra) — `chore/rename-vva-to-eca-infra` — cần `cdk diff` review, Owner duyệt.
- PR3: Phase 4 (docs + repo) — `chore/rename-vva-to-eca-docs` — sau khi PR1 merge.

Mỗi PR giữ `rg -i vva` giảm dần, không tăng.

---

## 9. Checklist Owner duyệt

- [ ] Tên đầy đủ ECA là gì? (`Embodied Conversational Agent` hay giữ `Embodied Conversational Agent`?)
- [ ] Tên repo mới? (`ECA` hay `Embodied-Conversational-Agent`?)
- [ ] Giữ `docs/archive/*` nguyên hay rename luôn?
- [ ] Chọn **A hay B** cho CFN stack rename (§R4)? Khuyến nghị **B** (tạo mới, cutover).
- [ ] Có cần `git filter-repo` lịch sử không? Khuyến nghị **không**.
- [ ] Thời gian xóa alias `VVA_*` / `/vva/`? Khuyến nghị 2–4 tuần sau deploy.

---

## 10. Tài liệu tham chiếu

- Audit 01/09: 725 match / 133 file (rg 15.1.0, git ls-files).
- `docs/worklogs/19-05-2026.md` ADR-001…005, `docs/plans/reupdate-plan.md` D1–D33.
- `docs/tracking/status.md` (Neon `eca_user` đã tồn tại), `infra/README.md` (SSM 6 param, deploy order).
- `agenticRAG/langgraph_agents/db/postgres.py` fallback warning, `infra/app.py:52` tag `Project: ECA`.

---

*K viết — N implement — Owner chốt tên & CFN strategy. Mọi đổi tên đều có alias 1 release, không silent-fallback.*
