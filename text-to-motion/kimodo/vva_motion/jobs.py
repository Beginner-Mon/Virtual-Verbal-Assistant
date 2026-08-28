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
from boto3.dynamodb.conditions import Attr, Key

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
    """PutItem có điều kiện. Trả 'created' hoặc 'exists' — dedupe là tính chất cấu trúc.

    The condition is the mirror of ``read_status``: **enqueue must be able to
    replace exactly the rows read_status refuses to return.** When it could
    not, an expired row was unreadable and unwritable at the same time — the
    node skipped its cache_hit branch, the PutItem was rejected, and it
    answered `queued` with a job id whose row still said `done`. Nothing
    queries `done`, so no worker ever picked it up and that prompt stayed
    un-renderable until the TTL sweeper arrived, which AWS only promises
    "within a few days". No error anywhere (ruling R27).

    One asymmetry on purpose: read_status also returns None for a row with no
    ``status``, and that case is NOT here. The only row it describes is the
    heartbeat — unexpired, and the row whose loss makes every kimodo_node call
    answer `unavailable`. A real job_id is 32 hex chars and cannot collide with
    ``worker#heartbeat``, so covering it would buy symmetry and risk the one
    row that must never be written over.

    A replacement reports 'created', not a third value: the caller's question
    is "is there a fresh queued row for me now", and the answer is yes.
    """
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
        table.put_item(
            Item=item,
            ConditionExpression=(
                "attribute_not_exists(job_id)"
                " OR attribute_not_exists(expires_at)"
                " OR expires_at < :now"
            ),
            ExpressionAttributeValues={":now": now},
        )
        return "created"
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return "exists"
        raise


def read_status(table, job_id: str) -> dict | None:
    """Đường ĐỌC, không ghi gì. Lease quá hạn được diễn giải thành failed ngay tại đây —
    không phụ thuộc heartbeat, vì worker treo thì heartbeat vẫn tươi.

    THREE ways this returns None, and each is a row that exists but must not be
    treated as a job:

    1. No item at all.

    2. Past ``expires_at``. The row existing is NOT proof it is valid, and the two
       clocks do not agree: asset_stack.py's S3 lifecycle rule deletes motions/*
       reliably at 24h, while DynamoDB TTL is a background sweeper AWS documents as
       running "within a few days", typically 48h. In that window the row still
       says `done` and the .bvh is already gone — kimodo_node would report a
       cache_hit and enqueue nothing, making that prompt permanently
       un-renderable, and motion_status would hand the browser a signed URL to a
       404. Checking here is the same rule shared/stm.py already enforces on its
       own reads; GetItem takes no filter expression, so it has to be application
       code. A row with no ``expires_at`` at all counts as expired: everything
       enqueue() writes has one, so its absence means the row was not written by
       this module.

    3. No ``status``. The heartbeat lives in this same table under the reserved
       key ``worker#heartbeat`` and carries only last_seen — it is not a job.
       That row is reachable from an authed public route
       (GET /motion/worker%23heartbeat), so reading it must be a plain
       not_found rather than a 500.
    """
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return None
    now = int(time.time())
    if int(item.get("expires_at", 0)) < now:
        return None
    status = item.get("status")
    if not status:
        return None
    if status == "processing" and int(item.get("lease_until", 0)) < now:
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
    """Expired rows are not depth. A worker that was off for a day leaves
    `queued` rows the TTL sweeper has not collected; counting them turns
    MAX_QUEUE_DEPTH into a lock, where the node answers `busy` to everyone
    while nothing is actually waiting. DynamoDB applies the filter before it
    counts, so this is a true post-filter count."""
    return table.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=Key("status").eq("queued"),
        FilterExpression=Attr("expires_at").gt(int(time.time())),
        Select="COUNT",
    )["Count"]


def claim_next_job(table) -> dict | None:
    """Lấy job 'queued' cũ nhất và đặt lease. ConditionExpression giữ bất biến
    'processing nghĩa là có người đang sở hữu' — và là khoá loại trừ sẵn sàng cho
    ngày chạy nhiều hơn một GPU.

    Expired rows are skipped: rendering a day-old request costs GPU seconds for
    an answer nobody is still waiting for, and a stale row sorts FIRST on this
    index (oldest created_at), so an unchecked loop picks it every single time.

    The skip is done here rather than as a FilterExpression because DynamoDB
    applies Limit BEFORE the filter — with a filtered query, 5 stale rows would
    return an empty page and hide a live job sitting behind them. Filtering the
    page in Python has the same blind spot past 5 rows, but at least the page
    is 5 real candidates.
    """
    now = int(time.time())
    rows = table.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=Key("status").eq("queued"),
        Limit=5,
    )["Items"]
    for row in rows:
        if int(row.get("expires_at", 0)) < now:
            continue
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
