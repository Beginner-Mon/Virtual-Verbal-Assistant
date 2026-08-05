---
date: 2026-08-05
tags: [mobile, capacitor, auth, cognito, app-links]
author: K
---

# Capacitor mobile auth — những việc chỉ Owner làm được

Phần code đã xong. Còn lại là hạ tầng: một domain, hai file association, và ba
nơi phải khai cùng một URL.

## Vì sao lại phải có domain thật

Amplify chọn redirect URI bằng cách **so origin của trình duyệt**, và hàm
`getRedirectUrl` của nó chỉ chấp nhận `http://` hoặc `https://`. Chạy thử chính
logic đó:

```
Web dev              -> http://localhost:5173/     ✅ đúng
Capacitor iOS        -> http://localhost:5173/     ❌ URL Vite trên điện thoại
Capacitor Android    -> http://localhost:5173/     ❌
Chỉ khai ecaapp://   -> THROW invalidRedirectException
```

Lý do dòng 2-3 sai: origin mặc định của Capacitor là `capacitor://localhost` /
`http://localhost`, mà Amplify kiểm bằng `redirect.includes(location.hostname)`
⇒ chuỗi `"localhost"` khớp với `http://localhost:5173/`.

Nên **custom scheme (`ecaapp://`) không dùng được**, và default của Capacitor
cũng không. Cách duy nhất không phải vá thư viện: cho app tự phục vụ từ **một
origin HTTPS thật**, rồi dùng **App Links (Android) / Universal Links (iOS)** để
hệ điều hành đưa callback về app.

## 1. Chọn host và khai vào 3 nơi

Giả sử `app.example.com`. **Cả ba phải giống hệt nhau**, lệch một dấu `/` là hỏng.

| Nơi | Giá trị |
|---|---|
| Biến môi trường lúc build | `MOBILE_APP_HOST=app.example.com`<br>`MOBILE_APP_ORIGIN=https://app.example.com` |
| Cognito callback/logout URLs | tự sinh từ `amplify/shared/origins.ts` |
| Google Cloud console → Authorized redirect URIs | `https://app.example.com/auth/callback/` |

> Google console là **thủ công**, Amplify không đụng tới được.

## 2. Hai file association phải nằm trên domain

### Android — `https://app.example.com/.well-known/assetlinks.json`

```json
[{
  "relation": ["delegate_permission/common.handle_all_urls"],
  "target": {
    "namespace": "android_app",
    "package_name": "com.eca.assistant",
    "sha256_cert_fingerprints": ["<SHA256 CỦA KEYSTORE KÝ APP>"]
  }
}]
```

Lấy fingerprint:
```bash
keytool -list -v -keystore <file>.keystore -alias <alias> | grep SHA256
```
⚠️ Bản debug và bản release **khác fingerprint**. Muốn test bản debug thì phải
khai cả hai, nếu không App Link sẽ im lặng không hoạt động.

### iOS — `https://app.example.com/.well-known/apple-app-site-association`

```json
{
  "applinks": {
    "details": [{
      "appID": "<TEAM_ID>.com.eca.assistant",
      "paths": ["/auth/callback/*"]
    }]
  }
}
```

Phải trả về `Content-Type: application/json`, **không** được redirect, **không**
có đuôi `.json` trong tên file.

## 3. Toolchain — máy hiện tại chưa có gì

| Cần | Trạng thái |
|---|---|
| JDK 17+ | ❌ chưa cài |
| Android Studio + SDK | ❌ chưa cài |
| Xcode + máy Mac | ❌ không thể trên Windows |

Sau khi cài xong:
```bash
cd ECA_UI/frontend
npm run build
npx cap add android
npx cap sync
npx cap open android
```

## 4. Thứ tự kiểm thử

1. Mở `https://app.example.com/.well-known/assetlinks.json` bằng trình duyệt —
   phải ra JSON, không phải trang 404 của SPA.
2. Cài app, mở link `https://app.example.com/auth/callback/?code=test` — **app
   phải mở lên**, không phải Chrome. Không được thì App Link chưa ăn, đừng test
   tiếp phần đăng nhập.
3. Đăng nhập Google trong app — trang Google phải mở ở **tab hệ thống**, không
   phải webview. Thấy `disallowed_useragent` nghĩa là `authSessionOpener` chưa
   được truyền vào.
4. Sau khi chọn tài khoản, app phải quay lại và **đã đăng nhập**.

## Đã làm sẵn trong code

- `amplify/shared/origins.ts` — gom 11 chỗ hardcode origin về một chỗ, đọc từ
  `MOBILE_APP_ORIGIN` / `WEB_APP_ORIGIN`.
- `amplify/functions/shared/cors.ts` — Lambda phản chiếu origin theo allowlist,
  kèm `Vary: Origin`. Trước đây hardcode `localhost:5173` nên app mobile bị
  **mọi endpoint** từ chối.
- `capacitor.config.ts` — `server.hostname` từ env, `androidScheme: 'https'`.
- `src/lib/nativeAuth.ts` — mở system browser, bắt `appUrlOpen`, ghi `code`/`state`
  vào URL của webview rồi replay listener của Amplify, và chuyển token sang
  Capacitor Preferences thay vì localStorage.
- `src/lib/googleSignIn.ts` — tự truyền `authSessionOpener` khi chạy native.

## Chưa làm

- **Chưa chạy thử trên máy thật lần nào** — thiếu toolchain và thiếu domain.
- Preferences **không phải secure storage**. Nó là app-private nhưng không được
  hardware-backed. Nên chuyển sang keystore/keychain trước khi có dữ liệu thật.
- `npx cap add android` chưa chạy ⇒ chưa có thư mục `android/`.
