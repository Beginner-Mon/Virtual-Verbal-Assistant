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
from decimal import Decimal

from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

TTL_SECONDS = 24 * 3600
LEASE_SECONDS = 120
MAX_RETRIES = 3

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


def _to_dynamo_number(value):
    """boto3's DynamoDB serializer raises TypeError on a raw Python `float`
    ("Float types are not supported. Use Decimal types instead."). Coerce here,
    at the one place every field passes through, rather than trusting each
    caller to remember — the next caller is production (kimodo_node), not a
    test that can be written around the bug.

    Via str(value) rather than Decimal(value) directly: Decimal(3.0) keeps
    float's binary imprecision (Decimal('3.000000000000000177...')); going
    through the repr string gives the clean Decimal('3.0') a human meant.
    """
    return Decimal(str(value)) if isinstance(value, float) else value


def enqueue(table, job_id: str, **fields) -> str:
    """PutItem có điều kiện. Trả 'created' hoặc 'exists' — dedupe là tính chất cấu trúc."""
    now = int(time.time())
    item = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "retry_count": 0,
        "expires_at": now + TTL_SECONDS,
        **{k: _to_dynamo_number(v) for k, v in fields.items()},
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


def claim_next_job(table) -> dict | None:
    """Lấy job 'queued' cũ nhất và đặt lease. ConditionExpression giữ bất biến
    'processing nghĩa là có người đang sở hữu' — và là khoá loại trừ sẵn sàng cho
    ngày chạy nhiều hơn một GPU."""
    rows = table.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=Key("status").eq("queued"),
        Limit=5,
    )["Items"]
    for row in rows:
        try:
            res = table.update_item(
                Key={"job_id": row["job_id"]},
                UpdateExpression="SET #s = :p, lease_until = :l",
                ConditionExpression="#s = :q",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":p": "processing", ":q": "queued",
                    ":l": int(time.time()) + LEASE_SECONDS,
                },
                ReturnValues="ALL_NEW",
            )
            return dict(res["Attributes"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
    return None


def complete_job(table, job_id: str, s3_key: str) -> None:
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET #s = :d, s3_key = :k",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":d": "done", ":k": s3_key},
    )


def fail_job(table, job_id: str, reason: str) -> None:
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET #s = :f, reason = :r",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":f": "failed", ":r": reason},
    )


def recover_abandoned_jobs(table) -> int:
    """Chạy MỘT LẦN lúc worker khởi động. Dọn row mà worker đời trước bỏ lại khi chết.

    Không phải để báo lỗi — read_status() đã lo phần user nhìn thấy. Hàm này tồn tại để
    job ĐƯỢC LÀM LẠI: nếu row kẹt 'processing' mãi thì PutItem có điều kiện sẽ luôn fail
    và động tác đó không bao giờ sinh được cho tới khi TTL 24h dọn đi.
    """
    now = int(time.time())
    rows = table.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=Key("status").eq("processing"),
    )["Items"]
    recovered = 0
    for row in rows:
        if int(row.get("lease_until", 0)) >= now:
            continue                                  # còn hạn: có người đang giữ
        tries = int(row.get("retry_count", 0)) + 1
        if tries > MAX_RETRIES:
            fail_job(table, row["job_id"], "too many retries")
        else:
            table.update_item(
                Key={"job_id": row["job_id"]},
                UpdateExpression="SET #s = :q, retry_count = :n REMOVE lease_until",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":q": "queued", ":n": tries},
            )
        recovered += 1
    return recovered
