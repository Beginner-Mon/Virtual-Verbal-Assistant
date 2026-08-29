"""The deployed image must contain the file it is told to run.

Dockerfile.prod shipped once without `COPY worker.py`. Nothing caught it:
`cdk synth` passed, the image built, ECR accepted it, the task started — and the
container exited immediately with `can't open file '/workspace/worker.py'`.
There is deliberately no ECS Service behind this task (kimodo_ecs_stack.py), so
nothing restarted it and nothing reported a crash loop; the visible symptom was
a g5.xlarge billing at full rate with an empty queue.

Docker is not available in CI (and building this image needs a GPU base layer
and several GB), so this is a static check of the build recipe rather than a
build. It is worth having anyway: the failure it guards is a one-line deletion
that reviews already missed once.

The image's own `RUN test -f /workspace/worker.py` is the real guard — it fails
the BUILD instead of a running instance. This test guards the guard.
"""
from pathlib import Path

import pytest

_KIMODO = Path("text-to-motion/kimodo")
_PROD = (_KIMODO / "Dockerfile.prod").read_text(encoding="utf-8")
_DEV = (_KIMODO / "Dockerfile").read_text(encoding="utf-8")


@pytest.mark.unit
def test_prod_image_copies_the_file_ecs_runs():
    """kimodo_ecs_stack.py sets command=["python", "worker.py"]."""
    assert "COPY worker.py /workspace/worker.py" in _PROD


@pytest.mark.unit
def test_prod_image_fails_the_build_when_the_entrypoint_is_missing():
    """`test -f` exits 1 when the file is absent, and a non-zero RUN aborts the
    build. Without this line the absence is only discovered on the GPU."""
    assert "RUN test -f /workspace/worker.py" in _PROD


@pytest.mark.unit
def test_prod_image_default_command_is_the_worker_not_the_mcp_server():
    """A container that starts mcp_server.py looks healthy and drains the queue
    never — the HTTP/MCP link is exactly what the DynamoDB queue replaced."""
    assert 'CMD ["python", "worker.py"]' in _PROD
    assert 'CMD ["python", "mcp_server.py"]' not in _PROD


@pytest.mark.unit
def test_dev_image_copies_the_worker_too():
    """Parity: the dev image is where the worker loop is exercised by hand."""
    assert "COPY worker.py /workspace/worker.py" in _DEV
    assert "RUN test -f /workspace/worker.py" in _DEV
