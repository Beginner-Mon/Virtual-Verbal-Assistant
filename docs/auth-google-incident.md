---
date: 2026-08-05
tags: [auth, cognito, google, oauth, incident]
author: K
---

# Đăng nhập Google trả về sai tài khoản, và không thoát ra được

## Kết luận

**Có cách xử lý. 3/5 lớp đã sửa xong và chạy được ngay.** Hai lớp còn lại chờ deploy.

> **Cập nhật 08/08** — Tri báo bản vá vẫn tạo `b@` trong Cognito + DynamoDB. Đúng.
> Đó là một lỗi **khác** với lỗi bản vá nhắm tới, và đã có cách sửa riêng:
> [mục bổ sung 08/08](#bổ-sung-0808--bản-vá-presignup-không-chặn-được-việc-tạo-b).

Về tranh luận nguyên nhân: **cả hai bên đúng một nửa.** Có **hai** phiên sống sót
trên hai domain khác nhau.

- Phiên khiến bị đẩy thẳng vào tài khoản sai *mà không hiện màn hình nào* là của
  **Cognito**.
- Phiên **Google** là lý do vì sao sửa xong lớp Cognito vẫn tái diễn.

Gốc rễ: **xoá localStorage bị coi là đăng xuất.**

---

## Bằng chứng

### 1. `src/components/AuthGuard.tsx:55` — toàn bộ xử lý khi lệch email

```ts
clearLocalAuthStorage()
window.location.replace('/login?error=email_mismatch')
```

Không có lời gọi nào tới Cognito. Không có đăng xuất.

### 2. `src/components/AuthGuard.tsx:7-26` — `clearLocalAuthStorage` làm gì

Duyệt `localStorage` + `sessionStorage`, xoá key có tiền tố
`CognitoIdentityServiceProvider…`, `amplify-…`, `com.amplify.…`.

Hai kho này thuộc **origin của app**. Không phải nơi Cognito lưu phiên.

### 3. AWS docs — Cognito có phiên riêng

> The `/logout` endpoint is a redirection endpoint. It signs out the user and
> redirects either to an authorized sign-out URL for your app client, or to the
> `/login` endpoint.

— `docs.aws.amazon.com/cognito/latest/developerguide/logout-endpoint.html`

Không gọi endpoint này thì người dùng **vẫn đang đăng nhập** phía Cognito.

### 4. AWS docs — `/logout` không chạm được Google

> The logout endpoint **doesn't sign users out of OIDC or social identity
> providers (IdPs)**. To sign users out from their session with an external IdP,
> direct them to the sign-out page for that provider.

— cùng trang trên

Đây là chỗ chẩn đoán "Google giữ session" **đúng**.

### 5. AWS docs — cách duy nhất buộc Google hỏi lại

> Amazon Cognito forwards all values of `prompt` except `none` to your IdPs…
> `prompt=select_account` … adds `prompt=select_account` to the URL path for the
> IdP redirect destination. When IdPs support this parameter, they request that
> users select the account that they want to log in with.

— `docs.aws.amazon.com/cognito/latest/developerguide/authorization-endpoint.html`

---

## Vì sao JavaScript không tự dọn được

Ba phiên nằm trên ba origin. Code của app chỉ với tới được cột trái.

| `localhost:5173` — ta với tới được | `…auth.amazoncognito.com` + `accounts.google.com` — không |
|---|---|
| localStorage | cookie phiên Cognito |
| sessionStorage | cookie phiên Google |
| token của Amplify | |

`clearLocalAuthStorage` chỉ dọn được cột trái. Đó là toàn bộ vấn đề.

---

## Diễn biến thật

1. Đang đăng nhập bằng email, bấm "Link Google" trong Profile.
2. Code gửi `prompt=select_account` → Google bày bảng chọn tài khoản.
3. Chọn nhầm `b@gmail.com`.
4. Cognito nhận identity mới, cấp token, **tạo phiên cho `b@`**.
5. App phát hiện email lệch → xoá localStorage → đá về `/login`.
6. **Phiên Cognito của `b@` vẫn còn nguyên.** Lần bấm Google kế tiếp, Cognito
   thấy phiên hợp lệ và cấp code ngay — không hỏi gì, không ghé qua Google.

### Ghi chú về bước 2

`prompt=select_account` **không phải cẩu thả**. Source Amplify:

```js
if (!input?.options?.prompt) await assertUserNotAuthenticated();
```

Luồng link chạy lúc đang đăng nhập. Bỏ `prompt` đi là ném
`UserAlreadyAuthenticatedException`. Bảng chọn tài khoản là tác dụng phụ buộc
phải chấp nhận.

---

## Đã sửa

| Lớp | Cách xử lý | Trạng thái |
|---|---|---|
| Storage app | `clearLocalAuthStorage()` — vốn đã có | giữ nguyên |
| Phiên Cognito | Chuyển hướng qua `/logout` của Cognito | ✅ chạy ngay |
| Phiên Google | Lần thử lại sau khi lệch ép `prompt=select_account` | ✅ chạy ngay |
| Tạo tài khoản trùng cho **cùng một người** | PreSignUp link identity vào user sẵn có | ⏸ chờ deploy |
| Tạo tài khoản cho **người khác** khi chọn nhầm | Bỏ hosted-UI khỏi luồng link (mục dưới) | ⏸ chờ deploy |

**Chi tiết dễ sập**: `logout_uri` phải khớp *tuyệt đối* với một URL đã đăng ký,
nên lý do lỗi gửi qua `sessionStorage` chứ không qua query string. `logout_uri`
dùng `/` — URL đã có sẵn trong pool đang chạy — nên bản vá **không cần deploy**.

---

## Hai lỗi còn lại

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| GitHub trong "Account Linked" | Thẻ tĩnh, nút "Link" không có `onClick`, vĩnh viễn "Not connected" | ✅ đã xoá |
| Google không hiện "Linked" | Đọc `custom:googleSub` — claim bơm từ dòng DynamoDB chỉ ghi ở vài đường. Đăng nhập bằng tài khoản đã link thì Cognito trả về user sẵn có, **không kích hoạt PostConfirmation** ⇒ không ai ghi | ✅ đổi sang claim `identities` |

`identities` do chính Cognito phát ra, liệt kê mọi provider gắn trên user ⇒ đúng
theo bản chất, không phụ thuộc bảng phụ.

---

---

## Bổ sung 08/08 — bản vá PreSignUp **không** chặn được việc tạo `b@`

Tri báo: chọn nhầm vẫn tạo `b@gmail.com` trong Cognito **và** trong DynamoDB.
Đọc lại code thì đúng, và bảng "Đã sửa" ở trên trước đây ghi gộp làm một dòng nên
dễ hiểu nhầm. **Đây là hai lỗi khác nhau:**

- Đợt 2 chống *trùng tài khoản cho cùng một người* — `a@` có sẵn email, link Google
  `a@` → vẫn 1 user. Cái này bản vá xử lý được.
- Bug của Tri là *tạo tài khoản cho người khác*. Bản vá **không** xử lý.

Đường đi, `functions/pre-sign-up-handler/handler.ts:226-228`:

```ts
const destination =
  candidates[0]?.Username ??
  (await createNativeAnchor(userPoolId, email, request.userAttributes));
```

Chọn nhầm `b@` → `findLinkableNativeUsers("b@")` trả 0 → `createNativeAnchor` gọi
`AdminCreateUser` → user `b@` ra đời. Dòng 235 `recordGoogleAvailable("b@")` ghi
tiếp row DynamoDB.

### Vì sao không sửa được **trong** PreSignUp

PreSignUp không phân biệt được "bấm nhầm lúc link" với "user Google mới đăng ký
lần đầu". Hai request giống hệt nhau: cùng app client, cùng provider, cùng
`triggerSource`, một địa chỉ chưa có tài khoản. Cognito không chuyển tiếp state
nào của ứng dụng để tách hai ca đó — không `state` của OAuth, không client
metadata. Mà ca thứ hai thì **phải** tạo tài khoản.

### Cách sửa: luồng link không đi qua Cognito nữa

| Trước | Sau |
|---|---|
| Profile → `signInWithRedirect` → hosted UI → PreSignUp → `AdminCreateUser` | Profile → Google Identity Services → ID token → `POST /api/user/link-google` |

`amplify/functions/link-google/handler.ts` verify ID token bằng JWKS của Google
(`aws-jwt-verify`, audience = client id, cả hai cách viết issuer), so email với
email của **chính người đang đăng nhập**, lệch thì **409** — chưa ghi gì. Khớp thì
gọi `AdminLinkProviderForUser`.

IAM của lambda này chỉ có đúng `AdminLinkProviderForUser`. **Không** có
`AdminCreateUser` — nó không tạo nổi user kể cả khi code sai.

Không có sign-up nào xảy ra ⇒ không có Cognito user, không có row DynamoDB.

Phụ thêm: `login_hint` **chạy thật** trên đường này. Cognito không forward được
sang Google (xem mục bằng chứng), còn GIS thì nhận trực tiếp — nên bảng chọn tài
khoản của Google mở sẵn đúng địa chỉ đang đăng nhập.

Hosted UI vẫn lo phần **đăng nhập** — chỗ mà một user Google mới thì *đáng* được
tạo tài khoản.

---

## Chưa chứng minh được

| Việc | Vì sao |
|---|---|
| Toàn bộ thay đổi backend auth | `npx ampx sandbox` chết ở `InvalidCredentialError: Failed to load default AWS credentials`. Chưa deploy lần nào |
| Chống tạo tài khoản trùng | Hệ quả dòng trên |
| `/api/user/link-google` | Hệ quả dòng trên. Cho tới khi deploy, **chọn nhầm vẫn tạo `b@`** — code mới chỉ `tsc` + `eslint` xanh |
| Bản vá phiên đăng nhập | Mới chỉ `tsc` + `npm run build` xanh. Phải bấm tay mới chắc |

Cần trước khi deploy: đặt `VITE_GOOGLE_CLIENT_ID` cho frontend (client id của
Google là giá trị công khai, không phải secret), và thêm origin của app vào
**Authorized JavaScript origins** trong Google Cloud console — GIS chặn theo
origin, khác với danh sách redirect URI của hosted UI.

---

## Cách kiểm chứng

### Luồng link (sau khi deploy)

1. Đăng nhập bằng email `a@…`
2. Profile → Account Linked → bấm nút Google → **cố ý chọn** `b@…`
3. Phải hiện lỗi ngay trong Profile: *"That Google account is b@…, but you are
   signed in as a@…"*. **Không** bị đá ra ngoài.
4. Kiểm tra Cognito: **không** được có user `b@`. Kiểm tra DynamoDB `UserMappings`:
   **không** được có row `b@`. Đây mới là phần cần nhìn tận mắt.
5. Bấm lại, chọn đúng `a@…` → badge đổi sang **Linked**.

### Luồng đăng nhập (đã chạy được, không cần deploy)

1. Đăng xuất, vào `/login`, "Continue with Google", chọn `b@…` khi app đang chờ `a@…`
2. Phải bị đá về `/login` kèm cảnh báo lệch tài khoản
3. Bấm lần nữa — **phải hiện lại bảng chọn tài khoản**. Nếu nó lẳng lặng vào thẳng
   `b@` thì bản vá chưa ăn.
