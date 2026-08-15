"""Direct, sandbox-only Stripe billing routes.

Billing in this service is deliberately unable to process real money:

* the feature is off unless ``BILLING_SANDBOX_ENABLED=true``;
* only Stripe test/restricted-test secret keys are accepted;
* live webhook events are rejected before any database write; and
* Stripe state never changes the internal ``FREE``/``DEMO`` entitlement.

Clerk is used only for authentication. Clerk Billing is not used here.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import stripe
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from langgraph_agents.api.auth import resolve_user_id
from langgraph_agents.shared import get_pg_client


router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutResponse(BaseModel):
    url: str


def _sandbox_enabled() -> bool:
    return os.getenv("BILLING_SANDBOX_ENABLED", "false").strip().lower() == "true"


def _test_key_configured() -> bool:
    key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    return key.startswith(("sk_test_", "rk_test_"))


def _webhook_configured() -> bool:
    return os.getenv("STRIPE_WEBHOOK_SECRET", "").strip().startswith("whsec_")


def _price_configured() -> bool:
    return os.getenv("STRIPE_TEST_PRICE_ID", "").strip().startswith("price_")


def _require_sandbox_enabled() -> None:
    if not _sandbox_enabled():
        raise HTTPException(503, "Stripe sandbox billing is disabled")


def _stripe_client() -> stripe.StripeClient:
    """Return a Stripe client only for an unmistakable sandbox key."""
    _require_sandbox_enabled()
    key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not key:
        raise HTTPException(503, "Stripe sandbox is not configured")
    if not key.startswith(("sk_test_", "rk_test_")):
        raise HTTPException(503, "Live Stripe keys are disabled by this application")
    return stripe.StripeClient(key)


def _stripe_v1(client: stripe.StripeClient) -> Any:
    """Support Stripe Python 12.0-14 while preferring the current v1 namespace."""
    return getattr(client, "v1", client)


def _test_price_id() -> str:
    price_id = os.getenv("STRIPE_TEST_PRICE_ID", "").strip()
    if not price_id.startswith("price_"):
        raise HTTPException(503, "STRIPE_TEST_PRICE_ID is not configured")
    return price_id


def _valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _app_url(request: Request) -> str:
    configured = os.getenv("FRONTEND_APP_URL", "").strip().rstrip("/")
    if configured:
        if not _valid_http_url(configured):
            raise HTTPException(503, "FRONTEND_APP_URL must be an absolute HTTP(S) URL")
        return configured

    # Origin is accepted as a convenience only for local development. A remote
    # deployment must explicitly configure FRONTEND_APP_URL, avoiding an
    # attacker-controlled Origin becoming a Stripe redirect destination.
    origin = request.headers.get("origin", "").strip().rstrip("/")
    parsed = urlparse(origin)
    if (
        _valid_http_url(origin)
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        return origin
    return "http://localhost:5173"


async def _ensure_demo_account(user_id: str) -> dict[str, Any]:
    pg = get_pg_client()
    await pg.connect()
    await pg.execute(
        "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING",
        user_id,
    )
    row = await pg.fetchrow(
        """INSERT INTO billing_accounts (user_id, access_plan)
           VALUES ($1::uuid, 'DEMO')
           ON CONFLICT (user_id) DO UPDATE SET updated_at = now()
           RETURNING user_id, access_plan, stripe_customer_id,
                     stripe_subscription_id, stripe_subscription_status,
                     updated_at""",
        user_id,
    )
    return dict(row)


def _object_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _reject_live_event(event: Any) -> None:
    if _object_value(event, "livemode", False) is not False:
        raise HTTPException(400, "Live Stripe webhook events are disabled")


@router.get("/config")
async def billing_config():
    fully_configured = (
        _test_key_configured() and _price_configured() and _webhook_configured()
    )
    return {
        "sandbox_enabled": _sandbox_enabled(),
        "test_only": True,
        "real_transactions_enabled": False,
        "clerk_billing_enabled": False,
        "stripe_configured": fully_configured,
        "checkout_enabled": _sandbox_enabled() and fully_configured,
        "webhook_configured": _webhook_configured(),
    }


@router.get("/status")
async def billing_status(request: Request, user_id: str = Query("anonymous")):
    uid = await resolve_user_id(request, user_id)
    row = await _ensure_demo_account(uid)
    return {
        "access_plan": row["access_plan"],
        "price": 0,
        "currency": "USD",
        "stripe_mode": "test",
        "real_transactions_enabled": False,
        "subscription_status": row["stripe_subscription_status"],
        "has_test_customer": bool(row["stripe_customer_id"]),
    }


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(request: Request, user_id: str = Query("anonymous")):
    uid = await resolve_user_id(request, user_id)
    account = await _ensure_demo_account(uid)
    stripe_client = _stripe_client()
    app_url = _app_url(request)

    params: dict[str, Any] = {
        "mode": "subscription",
        "line_items": [{"price": _test_price_id(), "quantity": 1}],
        "client_reference_id": uid,
        "success_url": f"{app_url}/?billing=sandbox-checkout-complete",
        "cancel_url": f"{app_url}/?billing=sandbox-checkout-cancelled",
        "metadata": {"vva_user_id": uid, "access_plan": "DEMO"},
        "subscription_data": {
            "metadata": {"vva_user_id": uid, "environment": "sandbox"}
        },
    }
    if account.get("stripe_customer_id"):
        params["customer"] = account["stripe_customer_id"]

    try:
        session = await _stripe_v1(stripe_client).checkout.sessions.create_async(params)
    except stripe.StripeError as exc:
        raise HTTPException(502, "Stripe sandbox checkout could not be created") from exc

    session_id = _object_value(session, "id", "")
    session_url = _object_value(session, "url")
    if (
        _object_value(session, "livemode", False) is not False
        or not str(session_id).startswith("cs_test_")
        or not session_url
    ):
        raise HTTPException(502, "Stripe returned a non-sandbox checkout session")
    return CheckoutResponse(url=session_url)


@router.post("/portal", response_model=CheckoutResponse)
async def create_portal(request: Request, user_id: str = Query("anonymous")):
    uid = await resolve_user_id(request, user_id)
    account = await _ensure_demo_account(uid)
    customer_id = account.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(409, "Complete a sandbox checkout first")

    try:
        session = await _stripe_v1(_stripe_client()).billing_portal.sessions.create_async(
            {
                "customer": customer_id,
                "return_url": f"{_app_url(request)}/?billing=sandbox-portal-returned",
            }
        )
    except stripe.StripeError as exc:
        raise HTTPException(502, "Stripe sandbox portal could not be created") from exc

    session_url = _object_value(session, "url")
    if not session_url:
        raise HTTPException(502, "Stripe sandbox did not return a portal URL")
    return CheckoutResponse(url=session_url)


@router.post("/webhook")
async def stripe_webhook(request: Request):
    _require_sandbox_enabled()
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret.startswith("whsec_"):
        raise HTTPException(503, "Stripe sandbox webhook is not configured")

    # Enforce the test key independently from signature verification. Stripe
    # webhook secrets do not encode whether they belong to live or test mode.
    _stripe_client()
    body = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(body, signature, secret)
    except (ValueError, stripe.SignatureVerificationError) as exc:
        raise HTTPException(400, "Invalid Stripe webhook signature") from exc

    _reject_live_event(event)
    event_id = str(_object_value(event, "id", ""))
    event_type = str(_object_value(event, "type", ""))
    if not event_id.startswith("evt_") or not event_type:
        raise HTTPException(400, "Invalid Stripe webhook event")

    data = _object_value(_object_value(event, "data", {}), "object", {})
    pg = get_pg_client()
    await pg.connect()

    if event_type == "checkout.session.completed":
        session_id = str(_object_value(data, "id", ""))
        if not session_id.startswith("cs_test_"):
            raise HTTPException(400, "Non-sandbox Checkout Session rejected")
        uid = _object_value(data, "client_reference_id")
        if uid:
            await _ensure_demo_account(uid)
            await pg.execute(
                """UPDATE billing_accounts
                   SET stripe_customer_id = $2,
                       stripe_subscription_id = $3,
                       stripe_subscription_status = 'sandbox_checkout_complete',
                       updated_at = now()
                   WHERE user_id = $1::uuid""",
                uid,
                _object_value(data, "customer"),
                _object_value(data, "subscription"),
            )
    elif event_type.startswith("customer.subscription."):
        metadata = _object_value(data, "metadata", {}) or {}
        uid = _object_value(metadata, "vva_user_id")
        if uid:
            await _ensure_demo_account(uid)
            await pg.execute(
                """UPDATE billing_accounts
                   SET stripe_customer_id = COALESCE($2, stripe_customer_id),
                       stripe_subscription_id = $3,
                       stripe_subscription_status = $4,
                       updated_at = now()
                   WHERE user_id = $1::uuid""",
                uid,
                _object_value(data, "customer"),
                _object_value(data, "id"),
                _object_value(data, "status", event_type),
            )

    # The state updates above are idempotent. Record the event after processing
    # so a transient database failure still allows Stripe to retry it.
    await pg.execute(
        """INSERT INTO billing_webhook_events (event_id, event_type, livemode)
           VALUES ($1, $2, false)
           ON CONFLICT (event_id) DO NOTHING""",
        event_id,
        event_type,
    )
    return {"received": True, "test_only": True}
