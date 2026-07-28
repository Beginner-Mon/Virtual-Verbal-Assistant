# Plan: ALB Endpoint cho Kimodo MCP Server

**Status:** Implemented (27/07/2026) — chờ N deploy.
**Ngày tạo:** 27/07/2026 — K (Mr. Senryuu)
**Ngườì implement:** N
**Phạm vi:** 1 file duy nhất — `infra/infra/kimodo_ecs_stack.py` (+ deploy/verify)
**Effort ước tính:** 1–2 giờ

---

## 1. Bối cảnh & Mục tiêu

Task Kimodo MCP chạy trên ECS (EC2 launch type, `g5.xlarge`, `network_mode=AWS_VPC`) không có endpoint ổn định: mỗi task có ENI riêng, IP đổi mỗi lần restart; gán public IP cho EC2 host không có tác dụng.

**Mục tiêu:** ALB internet-facing làm endpoint ổn định (`http://<alb-dns>/mcp`) để test MCP server từ bên ngoài. Thiết kế phải **dễ teardown/re-provision** (N sẽ cleanup sau khi test, nhưng ALB là thành phần chắc chắn dùng lại sau này).

**Xác nhận từ source** (`text-to-motion/kimodo/mcp_server.py`):
- FastMCP chạy `transport="streamable-http"`, port 8000 (env `MCP_PORT`).
- `health_check` ở line 337 là **MCP tool**, KHÔNG phải HTTP GET route → GET `/mcp` không kèm `Accept: text/event-stream` sẽ trả **406**. Điều này ảnh hưởng trực tiếp đến health check (xem §3.5).

## 2. Kiến trúc mục tiêu

```
Internet (IP của N) ──:80──> ALB (kimodo-alb, EcsPublic subnets, 2 AZ)
                               │ idle_timeout = 300s  (SSE)
                               ▼
                          Target Group (target_type=ip, :8000)
                               │ health check: GET /mcp, matcher 200-499
                               ▼
                     ECS Service kimodo-mcp-service (desired=1)
                               │ SG: chỉ nhận :8000 từ ALB SG + VPC CIDR
                               ▼
                     Task ENI (awsvpc) — kimodo-mcp container (1 GPU)
```

## 3. Quyết định thiết kế

| # | Quyết định | Lý do |
|---|---|---|
| 3.1 | **Cùng stack** `VvaKimodoEcsStack`, không tách stack mới | Service phải attach TG; tách stack tạo cross-stack dependency làm teardown rắc rối. Cleanup = toggle flag + deploy lại. |
| 3.2 | **Module constant `ENABLE_ALB = True`** ở đầu file, guard toàn bộ block ALB + service | Cleanup/re-provision = đổi 1 dòng + `cdk deploy`. Không cần comment/uncomment nhiều chỗ. |
| 3.3 | **HTTP :80, chưa làm HTTPS** | Phạm vi test. HTTPS (ACM + :443 + redirect) để cho production phase. |
| 3.4 | **Module constant `ALB_ALLOWED_CIDR = "0.0.0.0/0"`** kèm comment cảnh báo | MCP endpoint không có auth → **N phải đổi thành IP/32 của mình trước deploy** (và IP VPS sau này khi tích hợp orchestrator). Rẻ hơn và đơn giản hơn header-auth rule cho giai đoạn test. |
| 3.5 | **Sửa container health check**: `curl -fsS` → `curl -sS -o /dev/null` | GET `/mcp` trả 406 (FastMCP streamable-http) → `-f` exit 22 → container UNHEALTHY. Standalone task thì không sao, nhưng khi có service, ECS sẽ **giết task và thay thế liên tục (boot loop)**. Bỏ `-f`: bất kỳ HTTP response nào (kể cả 406) cũng chứng minh server đang listen. Không cần rebuild image. |
| 3.6 | **`health_check_grace_period = 600s`** trên service | Image kimodo nhiều GB — lần đầu pull trên instance mới + model load rất chậm. Thiếu grace period → ELB health check fail trong lúc khởi động → boot loop. |
| 3.7 | **TG health check matcher `200-499`** | Cùng lý do 406 ở §3.5. Không sửa app code. |
| 3.8 | **Giữ `min_healthy_percent=0, max_healthy_percent=100`** | g5.xlarge có đúng 1 GPU, task khai báo `gpu_count=1` → deploy mới bắt buộc phải kill task cũ trước. |
| 3.9 | **`deregistration_delay = 30s`** (default 300s) | Redeploy/test nhanh hơn. |
| 3.10 | **Capacity: `LaunchType=EC2` mặc định, KHÔNG chỉ định CP strategy** | CDK/AWS không yêu cầu sở hữu ASG — service đặt task lên mọi instance đã register vào cluster (ASG thủ công của N đã làm điều này). CP strategy tham chiếu bằng tên chỉ cần khi muốn managed scaling → để làm Option 2 (§4.4). Bối cảnh: service hiện tại được N tạo thủ công vì CDK version lúc đó chưa hỗ trợ CP ngoài; aws-cdk-lib 2.257.0 hiện tại đã hỗ trợ đầy đủ. |

## 4. Implementation spec (cho N)

File: `infra/infra/kimodo_ecs_stack.py`. Import thêm `aws_elasticloadbalancingv2 as elbv2`.

### 4.1 Constants đầu file

```python
ENABLE_ALB = True
ALB_ALLOWED_CIDR = "0.0.0.0/0"  # WARNING: đổi thành "<IP của N>/32" trước khi deploy
```

### 4.2 Security Groups

- **Mới** `kimodo-sg-alb`: ingress TCP 80 từ `ALB_ALLOWED_CIDR`, `allow_all_outbound=True`.
- **Sửa** `sg_ecs`: XÓA rule `any_ipv4():8000`, thay bằng:
  - TCP 8000 từ `sg_alb` — traffic qua ALB
  - TCP 8000 từ `10.0.0.0/16` (VPC CIDR) — giữ đường SSM port-forward fallback + truy cập nội bộ VPC

### 4.3 ALB + Target Group + Listener (trong `if ENABLE_ALB:`)

- `elbv2.ApplicationLoadBalancer`: `internet_facing=True`, `vpc_subnets=EcsPublic`, `security_group=sg_alb`.
  - `alb.set_attribute("idle_timeout.timeout_seconds", "300")` — SSE stream dài (default 60s sẽ giết stream).
- `elbv2.ApplicationTargetGroup`:
  - `vpc`, `port=8000`, `protocol=HTTP`, **`target_type=elbv2.TargetType.IP`** (bắt buộc cho awsvpc — đừng để default INSTANCE)
  - `health_check=elbv2.HealthCheck(path="/mcp", healthy_http_codes="200-499", interval=30s, timeout=10s, healthy_threshold_count=2, unhealthy_threshold_count=3)`
  - `tg.set_attribute("deregistration_delay.timeout_seconds", "30")`
- Listener: `alb.add_listener(port=80)`, default action `forward([tg])`.

### 4.4 Bật lại ECS Service (trong `if ENABLE_ALB:`)

Bỏ comment block `Ec2Service` hiện tại (dòng 166–178), giữ nguyên config, thêm:

```python
health_check_grace_period=Duration.seconds(600),
```

Sau đó: `service.attach_to_application_target_group(tg)`.

Giữ `assign_public_ip=True` (task vẫn cần đường ra IGW để pull ECR + đọc Secrets Manager).

**Capacity — chọn 1 trong 2:**

- **Option 1 (khuyến nghị cho test): giữ nguyên như block comment, KHÔNG thêm CP strategy** → service dùng `LaunchType=EC2`, đặt task lên bất kỳ instance nào đã register vào cluster. ASG/CP thủ công của N không cần CDK sở hữu. Đủ và đơn giản nhất cho test desired=1.
- **Option 2: tham chiếu capacity provider thủ công của N bằng tên** (giữ managed scaling/termination protection như setup hiện tại):
  ```python
  capacity_provider_strategies=[
      ecs.CapacityProviderStrategy(capacity_provider="<TÊN_CP>", weight=1),
  ],
  ```
  `<TÊN_CP>` lấy từ §5.0. Khi có CP strategy, CDK tự bỏ `LaunchType` (AWS cấm set cả hai). Cảnh báo: managed scaling có thể **scale-in (terminate) instance khi cluster trống task** nếu managed termination protection tắt → dễ bất ngờ trong lúc test.

### 4.5 Sửa container health check (dòng 135)

```python
command=["CMD-SHELL", "curl -sS -o /dev/null http://localhost:8000/mcp || exit 1"],
```

### 4.6 Outputs

- `AlbDnsName` = `alb.load_balancer_dns_name`
- Restore `ServiceName` output.

## 5. Pre-deploy checklist

0. **Đối chiếu trạng thái CFN ↔ AWS thực tế** (quyết định deploy có fail duplicate hay không):
   ```bash
   aws cloudformation list-stack-resources --stack-name VvaKimodoEcsStack \
     --query 'StackResourceSummaries[].[LogicalResourceId,ResourceType,ResourceStatus]' -o table
   aws ecs list-services --cluster kimodo-cluster
   aws ecs list-tasks --cluster kimodo-cluster
   ```
   - **Case A** — cluster/service/SG/log group đều có trong `list-stack-resources` (do stack này tạo từ deploy trước): an toàn, `cdk deploy` sẽ update in place, không duplicate.
   - **Case B** — service `kimodo-mcp-service` tồn tại trên AWS nhưng KHÔNG có trong stack (tạo thủ công): ECS `CreateService` **không idempotent** → deploy fail "service already exists" và rollback. Phải xóa service thủ công trước:
     ```bash
     aws ecs update-service --cluster kimodo-cluster --service kimodo-mcp-service --desired-count 0
     aws ecs delete-service --cluster kimodo-cluster --service kimodo-mcp-service
     ```
     CDK sẽ tạo lại service có quản lý + attach ALB. Không dùng CFN resource import — phức tạp, không đáng cho resource test.
   - Tương tự nếu SG `kimodo-sg-ecs` hoặc log group `/ecs/kimodo` tồn tại ngoài CFN → xóa trước khi deploy.
   - Kiểm tra capacity provider + default strategy của cluster:
     ```bash
     aws ecs describe-clusters --clusters kimodo-cluster --include ATTACHMENTS \
       --query 'clusters[].{CPs:capacityProviders, Default:defaultCapacityProviderStrategy}'
     ```
     Ghi lại tên CP nếu chọn Option 2 (§4.4). Nếu cluster có **default CP strategy**: service `LaunchType=EC2` sẽ override (AWS chỉ áp dụng default khi service không chỉ định launchType lẫn CP strategy) — nhưng nếu deploy báo lỗi CP conflict thì gỡ default:
     ```bash
     aws ecs put-cluster-capacity-providers --cluster kimodo-cluster \
       --capacity-providers <list CP hiện có> --default-capacity-provider-strategy []
     ```
1. **Đổi `ALB_ALLOWED_CIDR`** thành IP public/32 của N (`curl ifconfig.me`).
2. **Stop mọi task/service thủ công còn chiếm GPU** — g5.xlarge chỉ có 1 GPU; nếu không, service task mới sẽ PENDING vô hạn.
   `aws ecs list-tasks --cluster kimodo-cluster` → `aws ecs stop-task --task <arn>`
3. Xác nhận ASG `kimodo-asg` có ≥1 instance `InService` và registered vào cluster.
4. (Xác nhận giả định §3.5) Check health task hiện tại — kỳ vọng `UNHEALTHY`:
   `aws ecs describe-tasks --cluster kimodo-cluster --tasks <arn> --query 'tasks[].containers[].healthStatus'`
5. `cdk diff VvaKimodoEcsStack` — review kỹ: chỉ thêm ALB/TG/listener/service/SG rules, sửa health check; không động vào cluster/task-def/roles.

## 6. Deploy & Verify

Deploy: `cdk deploy VvaKimodoEcsStack`

Verify theo thứ tự:

1. **Target healthy** (có thể mất 5–10 phút nếu instance phải pull image mới):
   `aws elbv2 describe-target-health --target-group-arn <tg-arn>` → `state: healthy`
2. **Qua ALB được**: `curl -v http://<alb-dns>/mcp` → HTTP **406 là PASS** (nghĩa là request xuyên qua ALB tới server).
3. **MCP initialize**:
   ```bash
   curl -N -X POST http://<alb-dns>/mcp \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl-test","version":"0.1"}}}'
   ```
   → nhận SSE stream với `serverInfo` kimodo.
4. **Gọi tool `health_check`** qua MCP (tools/call) → xác nhận model status + GPU memory.
5. **SSE longevity**: giữ 1 stream > 60s để xác nhận `idle_timeout=300` có hiệu lực (trước đây sẽ bị cắt ở 60s).
6. (Tùy chọn) Trỏ MCP client thật vào `http://<alb-dns>/mcp`, gọi thử 1 motion generation ngắn.

## 7. Cleanup plan (sau khi test)

Theo thứ tự ưu tiên chi phí:

1. **Scale `kimodo-asg` về 0** — đây mới là chi phí lớn ($1.006/h ≈ **$24/ngày**), ASG tạo ngoài CDK nên phải làm thủ công.
2. `ENABLE_ALB = False` → `cdk deploy VvaKimodoEcsStack` — teardown có chọn lọc qua đường deploy bình thường: CFN xóa những resource đã rởi khỏi template (ALB/listener/TG/service, ~$0.75/ngày), giữ nguyên những resource còn trong template (cluster, task-def, SG, log group — $0).
3. **KHÔNG dùng `cdk destroy`** cho mục tiêu cleanup: destroy xóa TOÀN BỘ stack bất kể `ENABLE_ALB` (flag chỉ ảnh hưởng lúc synth template; destroy không quan tâm). Lưu ý: kể cả khi lỡ destroy, ASG/EC2 g5.xlarge/IAM roles/Secrets Manager đều nằm NGOÀI stack nên không bị động vào — nhưng cluster/task-def/SG/log group sẽ mất và phải provision lại ở lần deploy sau.
4. Dùng lại sau này: scale ASG lên 1 → `ENABLE_ALB = True` → deploy.

## 8. Chi phí

| Hạng mục | Đơn giá (us-east-1) | Ước tính |
|---|---|---|
| ALB cố định | $0.0225/giờ | ~$0.54/ngày |
| LCU (~1, traffic test thấp) | $0.008/LCU-giờ | ~$0.19/ngày |
| **ALB total** | | **~$0.75/ngày** (~$22/tháng nếu để 24/7) |
| g5.xlarge (tham chiếu) | $1.006/giờ | ~$24/ngày — **đây mới là thứ cần quản** |
| Data transfer | | negligible cho test |

## 9. Risks & Notes

| Risk | Mitigation |
|---|---|
| Deploy fail "service already exists" (service tạo thủ công ngoài CFN) | Pre-deploy bước 0: đối chiếu CFN ↔ AWS, `delete-service` trước (§5.0) |
| Deploy fail do SG `kimodo-sg-ecs` / log group `/ecs/kimodo` trùng tên ngoài CFN | Pre-deploy bước 0: xóa resource thủ công trùng tên trước |
| Managed scaling scale-in terminate instance khi cluster trống task (Option 2 + termination protection tắt) | Khuyến nghị Option 1 cho test; hoặc bật managed termination protection trên CP |
| Cluster default CP strategy conflict với service `LaunchType=EC2` | §5.0: kiểm tra bằng `describe-clusters`, gỡ default nếu deploy báo lỗi |
| Boot loop do container health check (`curl -f` + 406) | §3.5 — bắt buộc sửa cùng PR |
| Boot loop do image pull/model load chậm | `health_check_grace_period=600` (§3.6) |
| SSE bị cắt sau 60s | `idle_timeout=300` (§4.3) + verify bước 6.5 |
| Service task PENDING vô hạn do GPU bị chiếm | Pre-deploy bước 2: stop standalone task |
| ALB cần ≥2 AZ | Đã thỏa — VPC có 2 EcsPublic subnets (`max_azs=2`) |
| MCP endpoint public không auth | `ALB_ALLOWED_CIDR` restrict /32 (§3.4); production: HTTPS + auth sau |
| Tương lai scale >1 task | MCP session stateful theo `Mcp-Session-Id` → cần bật stickiness. Out of scope, ghi chú cho Phase 3 tích hợp. |

## 10. Sau khi implement

- N ghi worklog `docs/worklogs/27-07-2026.md`: kết quả deploy, verify, IP/dns endpoint.
- Cân nhắc bổ sung ADR nhỏ cho quyết định "ALB trước Kimodo MCP public endpoint" (hoặc gộp vào ghi chú ADR-005 revised — N quyết).
- K review worklog trước khi đóng task.
