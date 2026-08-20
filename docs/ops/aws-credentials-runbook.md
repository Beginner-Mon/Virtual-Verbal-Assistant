---
date: 2026-08-15
tags: [ops, aws, iam, amplify, credentials]
author: K
---

# Credentials cần cho `scripts/amplify_recover.sh`

## Trước hết: có thể không cần tạo gì

Tri đã deploy sandbox thành công ⇒ **Tri đã có credentials**. Rẻ nhất là Tri
`git pull` rồi tự chạy:

```bash
bash scripts/amplify_recover.sh status
```

Không tạo IAM user mới, không có access key nào tồn tại thêm trên đời. Chỉ tạo
credential khi cần chạy từ máy N.

---

## Hai mức quyền, cấp theo thứ tự

Đừng cấp mức 2 ngay. Mức 1 đủ để **chẩn đoán** và không hỏng được gì — chạy
`status` là biết stack chết vì lý do gì, rồi mới quyết có cần mức 2 không.

### Mức 1 — chỉ đọc (an toàn, cấp trước)

Đủ cho `status` và `secrets`. Không sửa, không xoá.

Về giá trị secret — nói chính xác: `ssm:GetParametersByPath` **có** trả về value,
nhưng SecureString trả về dạng đã mã hoá, và policy này **không cấp
`kms:Decrypt`** nên giá trị đó vô dụng. Script cũng chỉ in tên
(`--query 'Parameters[].Name'`). Không phải "API không cho đọc" — mà là "đọc ra
thì cũng không giải mã được".

`docs/ops/iam-vva-recover-readonly.json`

### Mức 2 — sửa được (chỉ cấp khi mức 1 đã chỉ ra việc phải làm)

Thêm: xoá stack, ghi secret, đổi env var, chạy deploy.

**`cloudformation:DeleteStack` là quyền nguy hiểm nhất ở đây.** CloudFormation
xoá stack bằng quyền của *chính người gọi*, nên principal phải xoá được mọi
resource bên trong: Cognito user pool (kèm **toàn bộ tài khoản trong đó**),
Lambda, DynamoDB, API Gateway, IAM role. Đó là lý do dùng managed policy
`AdministratorAccess-Amplify` thay vì liệt kê tay — liệt kê thiếu một action thì
stack kẹt ở `DELETE_FAILED`, tệ hơn lúc đầu.

`docs/ops/iam-vva-recover-fix.json` (phần bổ sung, dùng **kèm**
`AdministratorAccess-Amplify`)

---

## Tri làm cụ thể — ~5 phút

1. IAM Console → **Users** → *Create user* → tên `vva-recover-nguyen`
   → **không** bật console access
2. *Attach policies directly* → **Create policy** → tab **JSON** → dán nội dung
   `iam-vva-recover-readonly.json` → tên `VVARecoverReadOnly` → gắn vào user
3. User → tab **Security credentials** → *Create access key* → chọn
   **Command Line Interface (CLI)** → tải file `.csv`
4. Gửi file đó qua **trình quản lý mật khẩu** (1Password / Bitwarden).
   **Không** qua chat, Slack, Zalo, email.
5. Khi xong việc: IAM → user → *Deactivate* rồi *Delete* access key

Cần mức 2 thì lặp lại bước 2 với `iam-vva-recover-fix.json`, và gắn thêm managed
policy **`AdministratorAccess-Amplify`**.

## N làm — ~1 phút

```bash
aws configure
# AWS Access Key ID:     <từ file .csv>
# AWS Secret Access Key: <từ file .csv>
# Default region name:   us-east-1
# Default output format: json
```

`aws configure` hỏi ngay trong terminal. **Không dán key vào cửa sổ chat.**

Kiểm:

```bash
bash scripts/amplify_recover.sh status
```

---

## Nếu bị AccessDenied

Script không nuốt lỗi — AWS trả về tên action bị thiếu ngay trong thông báo, ví dụ:

```
User: arn:aws:iam::…:user/vva-recover-nguyen is not authorized to perform:
cloudformation:DescribeStackEvents
```

Thêm đúng action đó vào policy. Không đoán, không cấp `*` cho nhanh.

## Cách tốt hơn nếu team dùng lâu dài

IAM Identity Center (SSO) → `aws sso login --profile vva`. Credential hết hạn
theo phiên, không có access key vĩnh viễn nằm trên đĩa. Tốn thêm ~20 phút setup
lần đầu, đáng làm nếu còn phải deploy nhiều lần.
