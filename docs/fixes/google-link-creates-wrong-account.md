---
date: 2026-08-08
tags: [auth, cognito, google, oauth, fix, report]
author: K
to: Tri
branch: feature/langgraph-rewrite
---

# Báo cáo gửi Tri — chọn nhầm Google vẫn tạo `b@` trong Cognito + DynamoDB

## Trả lời thẳng

**Tri đúng. Code sửa đã xong. Chưa deploy nên chưa chứng minh được.**

Và đây là lỗi **khác** với lỗi bản vá cũ (Đợt 2) nhắm tới — chỗ này tôi ghi gộp
làm một dòng trong `auth-google-incident.md` nên dễ hiểu nhầm là đã xử lý:

| | Bug | Bản vá Đợt 2 |
| --- | --- | --- |
| A | Trùng tài khoản cho **cùng một người** (`a@` có email + link Google `a@`) | ✅ xử lý được |
| B | Tạo tài khoản cho **người khác** khi bấm nhầm (`b@`) | ❌ **không** xử lý |

Deploy Đợt 2 xong thì `b@` **vẫn bị tạo**. Cần bản vá riêng, đã viết hôm nay.

---

## Bằng chứng — đường đi tạo ra `b@`

`amplify/functions/pre-sign-up-handler/handler.ts:216-235`

```ts
const candidates = await findLinkableNativeUsers(userPoolId, email);   // 216
...
const destination =
  candidates[0]?.Username ??
  (await createNativeAnchor(userPoolId, email, request.userAttributes)); // 226-228
await linkGoogleTo(userPoolId, destination, googleSub);                  // 230
...
await recordGoogleAvailable(email, displayName);                        // 235
```

Bấm nhầm `b@`:

1. `findLinkableNativeUsers("b@")` → **0 candidate**
2. rơi vào `createNativeAnchor` → `AdminCreateUser` → **user `b@` ra đời trong Cognito**
3. dòng 235 `recordGoogleAvailable("b@")` → **row `b@` trong DynamoDB `UserMappings`**

Đúng cả hai chỗ Tri nói.

---

## Vì sao KHÔNG vá được trong PreSignUp

Đây là phần đáng đọc nhất, vì nó giải thích tại sao không phải chỉ thêm một câu
`if`.

PreSignUp không phân biệt được hai request này:

| | Bấm nhầm lúc link | User Google mới đăng ký lần đầu |
| --- | --- | --- |
| app client | giống nhau | giống nhau |
| provider | Google | Google |
| `triggerSource` | `PreSignUp_ExternalProvider` | `PreSignUp_ExternalProvider` |
| địa chỉ | chưa có tài khoản | chưa có tài khoản |
| **phải làm gì** | **từ chối** | **tạo tài khoản** |

Cognito không chuyển tiếp bất kỳ state nào của ứng dụng vào trigger — không
`state` của OAuth, không client metadata. Nên trong PreSignUp, hai ca đó là **một
request giống hệt nhau đòi hai kết quả ngược nhau**. Không có `if` nào viết được.

Mọi thứ chạy *sau* PreSignUp (PostConfirmation, pre-token-generation, check lệch
email ở frontend) đều vô dụng ở đây: lúc chúng chạy thì user đã tồn tại rồi.

---

## Cách sửa: luồng link không đi qua Cognito nữa

| Trước | Sau |
| --- | --- |
| Profile → `signInWithRedirect` → hosted UI → PreSignUp → `AdminCreateUser` | Profile → Google Identity Services → ID token → `POST /api/user/link-google` |

Qua hosted UI, "link" là một **sign-up** dưới mắt Cognito. Bỏ hosted UI khỏi luồng
link thì không còn sign-up nào để mà tạo nhầm.

### Lambda mới — `amplify/functions/link-google/handler.ts`

1. Verify ID token bằng JWKS của Google (`aws-jwt-verify`, audience = client id)
2. Bắt buộc `email_verified === true`
3. **So email của Google với email của chính người đang đăng nhập**
4. Lệch → **409 `EMAIL_MISMATCH`**, chưa ghi bất cứ đâu
5. Khớp → `AdminLinkProviderForUser`

### Điểm then chốt nằm ở IAM, không phải ở code

`amplify/backend.ts`:

```ts
(backend.linkGoogleHandler.resources.lambda as Function).addToRolePolicy(
  new PolicyStatement({
    actions: ['cognito-idp:AdminLinkProviderForUser'],
    resources: [userPoolArnDecoupled],
  }),
);
```

**Không có `AdminCreateUser`.** Lambda này không tạo nổi user kể cả khi code sai.
Đó là toàn bộ khác biệt so với đường hosted-UI nó thay thế — bảo đảm bằng quyền,
không bằng logic.

Không có sign-up ⇒ không Cognito user, không row DynamoDB.

### Ba quyết định phụ

| Quyết định | Lý do |
| --- | --- |
| Xoá hẳn option `alreadySignedIn` khỏi `googleSignIn.ts` | Không còn caller, và để không ai đi lại vào đường cũ. File đó giờ **chỉ** lo đăng nhập |
| Nút do Google render (`renderButton`), **không** One Tap (`prompt()`) | One Tap bị chặn bởi cài đặt trình duyệt / FedCM / lần dismiss trước — và khi bị chặn thì **im lặng** |
| Đăng ký **cả hai** cách viết issuer (`accounts.google.com` và dạng https) | Google phát cả hai. Pin một cái sẽ từ chối token thật lúc được lúc không — loại bug auth khó truy nhất |

Phụ thêm: `login_hint` **chạy thật** trên đường này. Cognito không forward được
sang Google (AWS docs nói thẳng — xem `auth-google-incident.md`); GIS nhận trực
tiếp. Nên bảng chọn của Google mở sẵn đúng địa chỉ đang đăng nhập.

Hosted UI vẫn lo phần **đăng nhập** — chỗ mà một user Google mới thì *đáng* được
tạo tài khoản.

---

## File

| Loại | File |
| --- | --- |
| Mới | `amplify/functions/link-google/handler.ts` |
| Mới | `src/lib/googleLink.ts` |
| Sửa | `amplify/backend.ts` (function + IAM + route `POST /api/user/link-google`) |
| Sửa | `src/components/ProfileContent.tsx` |
| Sửa | `src/lib/googleSignIn.ts` (xoá `alreadySignedIn`) |
| Dep | `aws-jwt-verify@5.2.1` |

---

## Trạng thái — tách bạch

### ✅ Xong

- `tsc -b` exit 0
- `tsc --noEmit -p amplify/tsconfig.json` exit 0
- eslint **0 lỗi** trên file mới (cố ý không copy kiểu `event: any` /
  `catch (error: any)` của các handler cũ)
- `npm run build` xanh, `npm test` 37/37

### 🔴 Chưa

**Chưa chạy thật lần nào.** `npx ampx sandbox` chết ở
`InvalidCredentialError: Failed to load default AWS credentials`.

**Tới khi deploy được thì chọn nhầm vẫn tạo `b@`.** Không nên hiểu bản báo cáo này
là "đã hết bug trên môi trường đang chạy".

Deploy cần **3** thứ, không phải 1:

1. `cd ECA_UI/frontend && npx ampx configure profile`
2. Đặt `VITE_GOOGLE_CLIENT_ID` cho frontend (client id Google là giá trị công khai,
   không phải secret)
3. Thêm origin của app vào **Authorized JavaScript origins** ở Google Cloud console
   — GIS chặn theo origin, **khác** danh sách redirect URI của hosted UI. Thiếu
   bước này thì nút Google không render và không có lỗi rõ ràng

---

## Cách Tri tự kiểm chứng sau khi deploy

1. Đăng nhập bằng email `a@…`
2. Profile → Account Linked → bấm nút Google → **cố ý chọn** `b@…`
3. Phải hiện lỗi ngay trong Profile: *"That Google account is b@…, but you are
   signed in as a@…"*. **Không** bị đá ra ngoài.
4. **Mở Cognito: không được có user `b@`. Mở DynamoDB `UserMappings`: không được có
   row `b@`.** ← đây mới là phần phải nhìn tận mắt, 3 bước trên chỉ là UI
5. Bấm lại, chọn đúng `a@…` → badge đổi sang **Linked**

Nếu bước 4 vẫn thấy `b@` thì bản vá chưa ăn — báo lại, đừng tin bước 3.

---

Liên quan: `docs/auth-google-incident.md` (bug phiên đăng nhập, đã sửa 3/5 lớp) ·
`docs/worklogs/08-08-2026.md` §5
