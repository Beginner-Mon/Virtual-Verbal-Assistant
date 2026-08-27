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

TTL_SECONDS = 24 * 3600
LEASE_SECONDS = 120


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
