"""Unit coverage for the sandbox-only billing safety boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from langgraph_agents.api import billing


@pytest.mark.unit
@pytest.mark.asyncio
async def test_config_is_disabled_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("BILLING_SANDBOX_ENABLED", raising=False)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_TEST_PRICE_ID", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)

    result = await billing.billing_config()

    assert result["sandbox_enabled"] is False
    assert result["checkout_enabled"] is False
    assert result["real_transactions_enabled"] is False
    assert result["clerk_billing_enabled"] is False


@pytest.mark.unit
def test_live_secret_key_is_rejected_even_when_sandbox_flag_is_enabled(monkeypatch):
    monkeypatch.setenv("BILLING_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_never_allowed")

    with pytest.raises(HTTPException) as exc_info:
        billing._stripe_client()

    assert exc_info.value.status_code == 503
    assert "Live Stripe keys are disabled" in exc_info.value.detail


@pytest.mark.unit
def test_live_webhook_event_is_rejected():
    with pytest.raises(HTTPException) as exc_info:
        billing._reject_live_event({"id": "evt_live", "livemode": True})

    assert exc_info.value.status_code == 400
    assert "Live Stripe webhook events are disabled" in exc_info.value.detail


@pytest.mark.unit
def test_remote_origin_cannot_control_checkout_redirect(monkeypatch):
    monkeypatch.delenv("FRONTEND_APP_URL", raising=False)
    request = SimpleNamespace(headers={"origin": "https://attacker.example"})

    assert billing._app_url(request) == "http://localhost:5173"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_checkout_uses_test_session_and_keeps_demo_metadata(monkeypatch):
    monkeypatch.setenv("STRIPE_TEST_PRICE_ID", "price_test_demo")
    create_async = AsyncMock(
        return_value=SimpleNamespace(
            id="cs_test_123",
            url="https://checkout.stripe.test/session",
            livemode=False,
        )
    )
    fake_client = SimpleNamespace(
        v1=SimpleNamespace(
            checkout=SimpleNamespace(
                sessions=SimpleNamespace(create_async=create_async),
            )
        )
    )
    user_id = "00000000-0000-0000-0000-000000000001"
    request = SimpleNamespace(headers={"origin": "http://localhost:5173"})

    # No auth patching: uid is now the handler's second parameter, supplied by
    # Depends(billing_user_id) at runtime and passed directly here.
    with (
        patch.object(
            billing,
            "_ensure_demo_account",
            AsyncMock(return_value={"access_plan": "DEMO", "stripe_customer_id": None}),
        ),
        patch.object(billing, "_stripe_client", return_value=fake_client),
    ):
        response = await billing.create_checkout(request, user_id)

    params = create_async.await_args.args[0]
    assert response.url == "https://checkout.stripe.test/session"
    assert params["mode"] == "subscription"
    assert params["metadata"]["access_plan"] == "DEMO"
    assert params["subscription_data"]["metadata"]["environment"] == "sandbox"
    assert params["line_items"] == [{"price": "price_test_demo", "quantity": 1}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_checkout_rejects_a_live_session_response(monkeypatch):
    monkeypatch.setenv("STRIPE_TEST_PRICE_ID", "price_test_demo")
    create_async = AsyncMock(
        return_value=SimpleNamespace(
            id="cs_live_123",
            url="https://checkout.stripe.com/session",
            livemode=True,
        )
    )
    fake_client = SimpleNamespace(
        v1=SimpleNamespace(
            checkout=SimpleNamespace(
                sessions=SimpleNamespace(create_async=create_async),
            )
        )
    )
    request = SimpleNamespace(headers={"origin": "http://localhost:5173"})

    with (
        patch.object(
            billing,
            "_ensure_demo_account",
            AsyncMock(return_value={"access_plan": "DEMO", "stripe_customer_id": None}),
        ),
        patch.object(billing, "_stripe_client", return_value=fake_client),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await billing.create_checkout(
                request, "00000000-0000-0000-0000-000000000001",
            )

    assert exc_info.value.status_code == 502
    assert "non-sandbox" in exc_info.value.detail


# ── The wiring, not the handlers ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/billing/status"), ("post", "/billing/checkout"), ("post", "/billing/portal")],
)
def test_billing_routes_reject_an_unauthenticated_call(method, path):
    """Exercise Depends(billing_user_id) rather than calling handlers with a uid.

    Every other test in this file passes the uid in as an argument, which skips
    the dependency entirely — so when billing_user_id was written as
    `(request: Request)` delegating to `current_user_id(request)`, the Request
    landed in the credentials parameter and every real call died with an
    AttributeError. The suite stayed green because billing_local.py overrides
    this dependency and the tests bypass it: the only unexercised path was
    production.

    No Cognito configuration is needed. A missing Authorization header is
    rejected before any token is verified, which is what makes this a unit test.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(billing.router)

    response = getattr(TestClient(app, raise_server_exceptions=False), method)(path)

    assert response.status_code == 401, (
        f"{method.upper()} {path} answered {response.status_code}. A 500 here means "
        f"the dependency raised instead of refusing; a 200 means it let an "
        f"anonymous caller through."
    )
