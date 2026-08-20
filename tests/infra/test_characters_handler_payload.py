"""The characters Lambda must route correctly under BOTH API Gateway payload formats.

It moved from a Lambda Function URL (payload format 2.0) to a REST API proxy
integration (format 1.0) on 20-08. The two formats put the method and the path in
different places, and every failure mode here is silent:

    format 2.0    event["rawPath"]                 event["requestContext"]["http"]["method"]
    format 1.0    event["path"]                    event["httpMethod"]
    format 1.0    event["requestContext"]["path"]  ← INCLUDES the stage. Never read this.

Reading `requestContext.path` under a REST API yields "/v1/characters", whose
first segment is "v1" rather than "characters", so the router 404s every request
while the function, the integration and the IAM policy all look correct.

These tests do not touch the database: they assert on the routing decision, which
is what the payload format governs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_LAMBDA_ROOT = Path(__file__).resolve().parents[2] / "infra" / "lambda"
sys.path.insert(0, str(_LAMBDA_ROOT / "layer"))
sys.path.insert(0, str(_LAMBDA_ROOT / "characters"))


@pytest.fixture(scope="module")
def handler_module():
    """Import the handler with its AWS/driver dependencies stubbed.

    The Lambda layer (boto3, pg8000) is built at deploy time and is not installed
    in the test environment. `pg8000.dbapi` has to be registered as its own
    sys.modules entry AND set as an attribute — `import pg8000.dbapi` needs both,
    and a bare MagicMock for "pg8000" satisfies neither.
    """
    pg8000 = MagicMock()
    pg8000.dbapi = MagicMock()
    sys.modules.setdefault("pg8000", pg8000)
    sys.modules.setdefault("pg8000.dbapi", pg8000.dbapi)
    sys.modules.setdefault("boto3", MagicMock())
    import handler
    return handler


def _v2(path: str, method: str = "GET", origin: str | None = None) -> dict:
    """A Lambda Function URL / HTTP API event."""
    event = {
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
        "headers": {},
    }
    if origin:
        event["headers"]["origin"] = origin
    return event


def _v1(path: str, method: str = "GET", stage: str = "v1", origin: str | None = None) -> dict:
    """A REST API proxy event.

    `path` excludes the stage; `requestContext.path` includes it. Both are
    populated exactly as API Gateway populates them, because the point of these
    tests is that the handler reads the right one.
    """
    event = {
        "path": path,
        "httpMethod": method,
        "requestContext": {"path": f"/{stage}{path}", "stage": stage},
        "headers": {},
    }
    if origin:
        event["headers"]["origin"] = origin
    return event


# ── Routing ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("make_event", [_v1, _v2], ids=["rest-1.0", "furl-2.0"])
@pytest.mark.parametrize(
    ("path", "expected_call"),
    [
        ("/characters", "_list_characters"),
        ("/characters/bronya", "_get_character"),
        ("/characters/bronya/avatar-profile", "_get_avatar_profile"),
    ],
)
def test_routes_the_same_under_both_payload_formats(
    handler_module, make_event, path, expected_call,
):
    with (
        patch.object(handler_module, "get_connection"),
        patch.object(handler_module, expected_call, return_value={"statusCode": 200}) as target,
    ):
        handler_module.handler(make_event(path), None)

    assert target.called, (
        f"{path} did not reach {expected_call} under this payload format"
    )


@pytest.mark.unit
def test_rest_api_stage_prefix_is_not_read(handler_module):
    """The specific mistake this file exists for.

    If the handler ever reads `requestContext.path`, the stage prefix makes the
    first segment "v1" and this returns 404 instead of listing characters.
    """
    with (
        patch.object(handler_module, "get_connection"),
        patch.object(handler_module, "_list_characters", return_value={"statusCode": 200}),
    ):
        response = handler_module.handler(_v1("/characters", stage="v1"), None)

    assert response["statusCode"] != 404, (
        "404 means the router saw the stage prefix — it is reading "
        "requestContext.path instead of event['path']"
    )


@pytest.mark.unit
@pytest.mark.parametrize("make_event", [_v1, _v2], ids=["rest-1.0", "furl-2.0"])
def test_non_read_methods_are_rejected_under_both_formats(handler_module, make_event):
    """A 405 here is why the gateway must answer OPTIONS itself.

    API Gateway's stage CORS configuration handles the preflight; if it did not,
    the OPTIONS would reach this function and be refused, and the browser would
    report a CORS failure rather than a 405.
    """
    response = handler_module.handler(make_event("/characters", method="POST"), None)
    assert response["statusCode"] == 405


# ── CORS ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_allowed_origin_is_echoed_not_wildcarded(handler_module, monkeypatch):
    """`*` was the effective policy until CloudFront stopped covering it.

    The wildcard lived in shared/response.py and was masked by CloudFront's
    ResponseHeadersPolicy (origin_override=True). Moving the catalog behind API
    Gateway removed that cover, so the value the function sends is now the value
    the browser sees.
    """
    from shared import response as response_module

    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com,http://localhost:5173")

    with (
        patch.object(handler_module, "get_connection"),
        patch.object(handler_module, "_list_characters",
                     side_effect=lambda cur: response_module.success({"characters": []})),
    ):
        result = handler_module.handler(
            _v1("/characters", origin="https://app.example.com"), None,
        )

    origin = result["headers"]["Access-Control-Allow-Origin"]
    assert origin == "https://app.example.com", f"expected the caller's origin, got {origin!r}"
    assert "Vary" in result["headers"], (
        "Vary: Origin is required once the header varies by caller, or a shared "
        "cache can hand one origin's response to another"
    )


@pytest.mark.unit
def test_unknown_origin_does_not_get_a_wildcard(handler_module, monkeypatch):
    from shared import response as response_module

    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")

    with (
        patch.object(handler_module, "get_connection"),
        patch.object(handler_module, "_list_characters",
                     side_effect=lambda cur: response_module.success({"characters": []})),
    ):
        result = handler_module.handler(
            _v1("/characters", origin="https://attacker.example"), None,
        )

    assert result["headers"]["Access-Control-Allow-Origin"] != "*"
    assert result["headers"]["Access-Control-Allow-Origin"] != "https://attacker.example"
