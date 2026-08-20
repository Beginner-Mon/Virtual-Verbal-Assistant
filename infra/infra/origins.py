"""The one CORS allow-list, shared by every stack that needs one.

Three stacks used to carry their own `_DEFAULT_ORIGINS`, and they had drifted:
AssetStack listed the Amplify release URL, CrudApiStack listed only localhost.
Both read the same `-c allowed_origins` context key, so they agreed whenever
someone passed the flag and disagreed whenever anyone forgot — which is the worst
shape a default can have, because the failure only appears in the environment
nobody is watching.

The deployed frontend has to be in the default, not only the dev origins. A list
that covers localhost and nothing else fails exactly once: in production, after
the release, with the browser reporting a blocked fetch and nothing pointing at
this file.
"""

from __future__ import annotations

DEFAULT_ALLOWED_ORIGINS = [
    "https://release.d32nf9wwqqt016.amplifyapp.com",
    "http://localhost:5173",
    # Browsers treat 127.0.0.1 and localhost as different origins, and Vite prints
    # whichever one you did not configure. Omitting this blocks every request —
    # including /health — while the server log stays empty, because the request
    # never leaves the browser.
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]


def resolve(node) -> list[str]:
    """Origins for a stack: `-c allowed_origins=a,b` if given, else the default.

    Pass `self.node` from inside a Stack.
    """
    raw = node.try_get_context("allowed_origins")
    parsed = [o.strip() for o in (raw or "").split(",") if o.strip()]
    return parsed or list(DEFAULT_ALLOWED_ORIGINS)
