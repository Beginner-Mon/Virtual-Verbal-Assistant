"""Motion job table — phiếu công việc giữa Lambda và worker Kimodo.

Không có endpoint nào giữa hai bên: Lambda ghi row, worker poll row. Xem phần
thiết kế trong plan để biết vì sao không dùng SQS (task ENI chỉ với tới S3 và
DynamoDB qua gateway endpoint miễn phí).
"""
from __future__ import annotations

import hashlib
import hmac
import time
import unicodedata

from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

TTL_SECONDS = 24 * 3600
LEASE_SECONDS = 120

HEARTBEAT_KEY = "worker#heartbeat"
HEARTBEAT_SECONDS = 30
HEARTBEAT_DEAD_AFTER = 90
HEARTBEAT_TTL = 300
MAX_QUEUE_DEPTH = 20
SECONDS_PER_JOB = 5


def canonical_request(prompt: str, duration: float, steps: int, model: str) -> str:
    """Chuẩn hoá để hai câu chỉ khác hoa/thường/khoảng trắng ra cùng một job."""
    text = unicodedata.normalize("NFC", prompt).strip().lower()
    text = " ".join(text.split())
    return f"{text}|{duration:.2f}|{steps}|{model}"


def compute_job_id(secret: str, prompt: str, duration: float, steps: int, model: str) -> str:
    """HMAC chứ không phải sha256 trần: cùng nội dung vẫn ra cùng key (cache còn trúng),
    nhưng người ngoài không tính được key từ prompt đoán được."""
    msg = canonical_request(prompt, duration, steps, model).encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:32]


def enqueue(table, job_id: str, **fields) -> str:
    """PutItem có điều kiện. Trả 'created' hoặc 'exists' — dedupe là tính chất cấu trúc."""
    now = int(time.time())
    item = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "retry_count": 0,
        "expires_at": now + TTL_SECONDS,
        **fields,
    }
    try:
        table.put_item(Item=item, ConditionExpression="attribute_not_exists(job_id)")
        return "created"
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return "exists"
        raise


def read_status(table, job_id: str) -> dict | None:
    """Đường ĐỌC, không ghi gì. Lease quá hạn được diễn giải thành failed ngay tại đây —
    không phụ thuộc heartbeat, vì worker treo thì heartbeat vẫn tươi."""
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return None
    if item["status"] == "processing" and int(item.get("lease_until", 0)) < int(time.time()):
        return {**item, "status": "failed", "reason": "lease expired"}
    return dict(item)


def write_heartbeat(table) -> None:
    """Worker ghi, Lambda đọc. Chỉ mang last_seen — không mang trạng thái job."""
    now = int(time.time())
    table.put_item(Item={
        "job_id": HEARTBEAT_KEY, "last_seen": now, "expires_at": now + HEARTBEAT_TTL,
    })


def worker_alive(table) -> bool:
    """So last_seen, KHÔNG dựa vào việc row còn tồn tại: TTL của DynamoDB có thể trễ tới 48h."""
    item = table.get_item(Key={"job_id": HEARTBEAT_KEY}).get("Item")
    if item is None:
        return False
    return int(time.time()) - int(item["last_seen"]) < HEARTBEAT_DEAD_AFTER


def queue_depth(table) -> int:
    return table.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=Key("status").eq("queued"),
        Select="COUNT",
    )["Count"]
