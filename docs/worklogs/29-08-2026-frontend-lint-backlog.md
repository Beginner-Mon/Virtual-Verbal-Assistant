---
date: 2026-08-29
tags: [handoff, frontend, lint, ci, backlog]
author: K
branch: feature/motion-frontend
---

# Bàn giao: 62 lỗi lint đang làm đỏ job `Frontend Build`

Viết để một phiên khác xử lý. **Không phải việc gấp** — nó đỏ từ trước nhánh
này, không chặn deploy, và Amplify vẫn build production bình thường.

## Vì sao job đỏ

`.github/workflows/release-tests.yml`, job `🏗️ Frontend Build`:

```
npm ci  →  npm run lint  →  npm run build
```

`npm run lint` đỏ nên **build không bao giờ chạy**. Muốn job xanh phải hết
**toàn bộ** 62 lỗi — sửa một phần không đổi màu.

Đo lúc bàn giao: `tsc --noEmit` **sạch**, `vitest` **109/109 xanh**,
`vite build` **thành công**. Chỉ lint đỏ.

## Đã sửa (74 → 62)

Commit `b3a83c5`.

**Một bug thật, không phải chuyện hình thức.** `ProfileSettingsModal.tsx:53` có
`onBack ?? onClose()`. Biểu thức này *đọc* `onBack` rồi vứt giá trị đi: khi
component cha truyền `onBack`, kết quả là chính hàm đó và **không gì được gọi** —
nút quay lại chết. Chỉ khi `onBack` vắng mặt thì `onClose()` mới chạy, nên nhìn
qua tưởng hoạt động. Sửa thành `(onBack ?? onClose)()`.

Ngoài ra: thay `getNormalizedBoneNode('hips' as any)` bằng
`VRMHumanBoneName.Hips` (đã export sẵn, `BodyPartPicker` dùng rồi), và
`restPoses as Map<string, any>` bằng `RestPoseMap` mới (`src/lib/restPose.ts`)
— hình dạng thật ghi ở `CharacterViewer.tsx:95`. Bỏ hai import chết giữ lại
"reserved for auth feature" sau `@ts-expect-error`.

## 62 lỗi còn lại

### 32 — `@typescript-eslint/no-explicit-any`

| Số | Nơi | Ghi chú |
| --- | --- | --- |
| 15 | `amplify/functions/*` (6 handler + `shared/cors.ts`) | **Không phải frontend** — Cognito Lambda, deployment riêng. `@types/aws-lambda` đã cài, dùng `PreSignUpTriggerEvent` v.v. Sai kiểu ở đây hỏng luồng đăng nhập. |
| 13 | `catch (err: any)` rải ở 6 file auth | Bỏ `: any` thì TS cho `unknown`, mọi chỗ đọc `err.message` sẽ đỏ. Cần một helper `errorMessage(e: unknown)` dùng chung. |
| 4 | `AuthGuard`, `ProfileContent`, `CharacterViewer`, `amplify-env.d.ts` | `useState<any>` cho session Amplify, `useRef<any>` cho OrbitControls. |

Nhóm này **an toàn về hành vi** nhưng phần lớn nằm trong đường đăng nhập, nên
phải thử login thật sau khi sửa.

### 24 — nhóm `react-hooks/*` (RỦI RO)

```
11  react-hooks/refs              ProfileContent 4, ChatContext 3, CharacterViewer 2, …
 5  react-hooks/immutability      CharacterViewer 3, ChatMessage 1, RendererSetup 1
 5  react-hooks/set-state-in-effect  ChatMessage, FloatingNavBar, ProfileSettingsModal,
                                     ChatContext, LoginPage
 3  react-hooks/purity            CharacterViewer 3
```

Đây là bộ luật thời React Compiler. Sửa cho **đúng** thường phải cấu trúc lại
effect/ref, tức **đổi hành vi thật** — không phải đổi kiểu. Và chúng tập trung ở
`CharacterViewer`, `ChatContext`, `ProfileContent` — code vừa được sửa trong
`feature/frontend-fixes`.

**Đừng sửa hàng loạt.** Từng cái một, chạy `vitest` + thử tay sau mỗi cái.

### 6 — `react-refresh/only-export-components`

`ChatContext`, `MotionContext`, `GraphicsContext`, `ThemeContext`,
`components/ui/button.tsx`.

Mỗi file vừa export component (`XProvider`) vừa export hook/context (`useX`).
Sửa = tách context + hook sang file riêng.

**Đáng làm nhất trong ba nhóm**, vì nó chữa một cái đau có thật: đây chính là lý
do sửa `ChatContext.tsx` xong phải hard-reload, và trong phiên này nó đã gây ra
`Uncaught Error: useChat must be used within ChatProvider` — mất khá nhiều thời
gian chẩn đoán (cuối cùng phải xoá `node_modules/.vite` và chạy `vite --force`).

## Gợi ý thứ tự

1. **`react-refresh` (6)** — cơ học, không đổi hành vi, và chữa dứt lỗi HMR.
2. **`no-explicit-any` (32)** — chia hai đợt: `amplify/functions` riêng (thử
   login), phần `src` riêng.
3. **`react-hooks` (24)** — cuối cùng, từng cái, có kiểm chứng.

Chỉ sau bước 3 thì `Frontend Build` mới xanh.

## Cảnh báo

Job này đã đỏ đủ lâu để không ai còn nhìn. Đó là rủi ro thật: một lỗi mới sẽ
lẫn vào và không ai thấy. Nếu chưa định sửa hết trong thời gian gần, cân nhắc
tách `npm run lint` thành job riêng **không chặn** build, để build lấy lại tín
hiệu thật — nhưng đó là quyết định, không phải mặc định.
