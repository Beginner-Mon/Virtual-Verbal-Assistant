"""Decide STREAM vs BUFFERED by arrival timing, not by eye.

    python infra/spike/verify_stream.py https://xxxx.execute-api.us-east-1.amazonaws.com/v1/probe

`curl -N` is not a test. A buffered response that arrives quickly and a streamed
one that arrives slowly look the same in a terminal, and the difference between
"working" and "looks slightly slow" is exactly the failure mode this spike
exists to rule out — see the Phase C acceptance note in the plan.

So: record when each SSE event ARRIVES and compare the spread against the server
timestamps the probe embeds.

    STREAMED   arrivals spread across ~5s, tracking emitted_at
    BUFFERED   all ten land within milliseconds of each other, at the end

Uses urllib rather than requests/httpx so the script runs against any Python
with no install — and because some HTTP clients buffer the whole body
themselves, which would make the client the thing being measured.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

# A buffered response delivers everything in one burst. Real streaming across a
# 5-second emit window puts seconds between first and last. Anything under half
# a second is a burst by any reading; the observed values are not close to this
# line in either direction, so its exact placement is not load-bearing.
STREAM_SPREAD_FLOOR_SECONDS = 0.5


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    url = sys.argv[1]

    print(f"[verify] GET {url}")
    started = time.monotonic()
    arrivals: list[tuple[float, dict]] = []

    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request, timeout=120) as response:
        print(f"[verify] status {response.status}, "
              f"content-type {response.headers.get('Content-Type')}")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            at = time.monotonic() - started
            try:
                payload = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            arrivals.append((at, payload))
            label = payload.get("seq", payload.get("total", "?"))
            emitted = payload.get("emitted_at")
            print(f"  arrived {at:6.3f}s   seq={label}"
                  + (f"   server emitted_at={emitted:.3f}s" if emitted is not None else ""))

    if len(arrivals) < 2:
        print(f"[verify] only {len(arrivals)} events — cannot judge")
        return 2

    spread = arrivals[-1][0] - arrivals[0][0]
    print()
    print(f"[verify] {len(arrivals)} events, first at {arrivals[0][0]:.3f}s, "
          f"last at {arrivals[-1][0]:.3f}s, spread {spread:.3f}s")

    if spread >= STREAM_SPREAD_FLOOR_SECONDS:
        print("[verify] STREAMED — API Gateway passed chunks through as they were "
              "produced. LWA's prelude satisfies Lambda-proxy STREAM mode.")
        return 0

    print("[verify] BUFFERED — every event landed at once. API Gateway waited for "
          "the whole body.")
    print("         Check, in order: responseTransferMode is STREAM on the "
          "integration; AWS_LWA_INVOKE_MODE=response_stream on the function; "
          "the endpoint is REGIONAL (edge-optimized has a 30s idle cap and "
          "different behaviour).")
    print("         If all three are right, LWA's prelude does NOT match what "
          "API Gateway expects — fall back to HTTP proxy + Function URL "
          "(plan bản 1) and budget for the shared secret.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
