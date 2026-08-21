"""Streaming probe — answers one question and nothing else.

    Does the AWS Lambda Web Adapter, in response_stream mode, emit the response
    format that API Gateway's Lambda-proxy STREAM integration expects?

AWS documents LWA as the supported way to stream from Python Lambdas, and
documents API Gateway's Lambda-proxy STREAM mode as requiring "a specific
response format" — a prelude carrying status code and headers. What no document
states is that LWA's prelude is that prelude: every LWA streaming example is
written against Function URLs. The whole /chat design rests on the two agreeing,
so it gets tested before anything is built on it.

Deliberately NOT a container. LWA is the same binary either way and the invoke
mode is an environment variable, so a zip answers the same question — while a
container would fold in a second unknown (does the 1.2 GB image bake the model
correctly?) and leave a failure ambiguous between the two.

The probe emits ten events half a second apart, each stamped with the server
time it was written. The client (verify_stream.py) records when each one
ARRIVES. Buffered and streamed responses are then trivially distinguishable:

    streamed   arrivals spread over ~5s, tracking the emit stamps
    buffered   all ten arrive within milliseconds of each other, at the end

Timing rather than eyeballing `curl -N`, because a fast buffered response and a
slow streamed one look identical to a human watching a terminal.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

EVENT_COUNT = int(os.getenv("PROBE_EVENTS", "10"))
EVENT_GAP_SECONDS = float(os.getenv("PROBE_GAP_SECONDS", "0.5"))

app = FastAPI(title="LWA streaming probe")


@app.get("/health")
async def health():
    """Readiness for LWA (AWS_LWA_READINESS_CHECK_PATH).

    Without it LWA probes "/" and waits — the same trap crud_api_stack.py
    documents.
    """
    return {"status": "ok"}


@app.get("/probe")
async def probe():
    async def generate():
        start = time.time()
        for i in range(EVENT_COUNT):
            payload = {
                "seq": i,
                "emitted_at": round(time.time() - start, 3),
            }
            yield f"event: tick\ndata: {json.dumps(payload)}\n\n"
            await asyncio.sleep(EVENT_GAP_SECONDS)
        yield f"event: done\ndata: {json.dumps({'total': EVENT_COUNT})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            # Belt and braces: some proxies buffer text/event-stream unless told
            # not to. API Gateway is not one of them, but if the probe fails it
            # should fail for the reason under test and not for this.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
