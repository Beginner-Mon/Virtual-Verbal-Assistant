"""The two data-deleting routes must stay off unless explicitly switched on.

`api/main.py` defines DELETE /me and DELETE /sessions/{sid}/messages/{mid}.
Until 21-08 that module only ever ran on a developer machine, so whether the
routes were reachable was academic. Deploying the agent to Lambda ends that: an
ungated `create_app()` publishes account deletion the same day /chat ships.

Owner deferred account deletion on 21-08 to reconsider its security, and the
reason is not cosmetic — `db/gdpr.py::delete_user` deletes PostgreSQL rows and
nothing else. The Cognito user survives, so the person can still sign in and
`routes_crud.py` recreates their `users` row on the next write; the DynamoDB
UserMappings row survives too, so re-registering with the same email links the
new sign-up to the OLD identity. Shipping that under the name "delete my
account" would be telling users something untrue.

These tests exist so the gate cannot be removed by accident. They assert the
DEFAULT, not just the mechanism: a gate that defaults to open protects nothing,
and there is no per-user backup to undo a mistake with.

See docs/plans/langgraph-agent-hosting.md §7 and docs/tracking/tech-debt.md.
"""

from __future__ import annotations

import pytest


_GDPR_ROUTES = {
    "/me",
    "/sessions/{session_id}/messages/{message_id}",
}


def _route_paths(app) -> set[str]:
    """DELETE paths the app actually exposes."""
    return {
        route.path
        for route in app.routes
        # `or set()` because Mount and WebSocketRoute have no `methods` at all,
        # and a Route can carry None — `in None` raises rather than returning
        # False, which would fail the test for the wrong reason.
        if "DELETE" in (getattr(route, "methods", None) or set())
    }


@pytest.mark.unit
def test_gdpr_routes_absent_by_default(monkeypatch):
    """No env var set at all — the routes must not exist.

    `delenv` rather than setting "false": the question is what a deployment that
    says NOTHING about this flag gets, because that is what a forgotten
    environment variable looks like.
    """
    monkeypatch.delenv("ENABLE_GDPR_ROUTES", raising=False)

    from langgraph_agents.api.main import create_app

    paths = _route_paths(create_app())
    assert not (_GDPR_ROUTES & paths), (
        f"Account-deletion routes are exposed with ENABLE_GDPR_ROUTES unset: "
        f"{sorted(_GDPR_ROUTES & paths)}. The default must be off — a deploy "
        f"that forgets the variable must not publish an irreversible delete."
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", ["false", "0", "no", "", "True ", "yes", "1"])
def test_only_the_exact_string_true_opens_the_gate(monkeypatch, value):
    """Anything that is not "true" (case-insensitive, trimmed) keeps it shut.

    "1" and "yes" are in this list on purpose. They read as affirmative to a
    human, and a gate that accepts them has a wider surface than the one
    documented — someone sets ENABLE_GDPR_ROUTES=1 believing it does nothing.
    Note "True " (trailing space) IS accepted, because the check trims and
    lowercases; it is here to pin that as deliberate rather than accidental.
    """
    monkeypatch.setenv("ENABLE_GDPR_ROUTES", value)

    from langgraph_agents.api.main import create_app

    paths = _route_paths(create_app())
    opened = bool(_GDPR_ROUTES & paths)
    expected = value.strip().lower() == "true"

    assert opened is expected, (
        f"ENABLE_GDPR_ROUTES={value!r} produced routes={opened}, expected "
        f"{expected}. Only the exact string 'true' may open a route that "
        f"deletes user data."
    )


@pytest.mark.unit
def test_gdpr_routes_present_when_enabled(monkeypatch):
    """The gate must actually be openable — otherwise it is dead code.

    Without this, the two tests above would still pass if someone deleted the
    routes entirely, and the suite would report health while the feature was
    gone.
    """
    monkeypatch.setenv("ENABLE_GDPR_ROUTES", "true")

    from langgraph_agents.api.main import create_app

    paths = _route_paths(create_app())
    missing = _GDPR_ROUTES - paths
    assert not missing, (
        f"ENABLE_GDPR_ROUTES=true but these routes are missing: {sorted(missing)}. "
        f"Either they were deleted, or their paths changed and this test's "
        f"_GDPR_ROUTES no longer names them."
    )
