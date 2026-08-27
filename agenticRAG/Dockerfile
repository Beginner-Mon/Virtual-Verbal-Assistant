# vva-agent — LangGraph + /chat, as a Lambda container image.
#
# Built in CI, never locally (Owner 21-08). See .github/workflows/deploy-agent.yml.
#
#   docker build -f agenticRAG/Dockerfile -t vva-agent .     # context = REPO ROOT
#
# Four decisions here are load-bearing, and three of them fail at INIT — inside a
# 10-second budget, visible only in CloudWatch, on a 10-minute CI loop. They are
# spelled out rather than left to be rediscovered.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — export the ONNX model, then throw this whole stage away.
#
# torch and optimum live ONLY here. That is the entire point of the multi-stage
# build: the agent needs e5-small's weights, not the framework that produced
# them. Collapsing this into one stage leaves ~2 GB of PyTorch in the image and
# every cost argument for ONNX evaporates — billing is memory x wall-clock, so
# the ~20s a turn spends waiting on DeepSeek would be 20s of paying for a
# PyTorch that does nothing.
#
# `optimum` AND `optimum-onnx`: optimum 2.x moved the exporter into a second
# distribution, so `optimum[exporters]` alone yields an optimum-cli with no
# `export onnx` subcommand — an error that reads like a broken install.
# ─────────────────────────────────────────────────────────────────────────────
FROM public.ecr.aws/docker/library/python:3.12-slim AS export

# `onnx` and `onnxruntime` are NOT optional here, and optimum-onnx does not pull
# them. Without them the export runs, traces the model, and then dies at the last
# step:
#
#     RuntimeError: The onnx and onnxruntime packages are necessary to fix the
#     dynamic shapes of the exported model.
#
# It cost a CI build to learn, and it should not have: the local export "worked"
# only because firstconda already had onnxruntime installed for unrelated
# reasons. Verifying a build step against a polluted environment is the same
# mistake as testing an image against the full requirements file — which is why
# this line was reproduced in a clean venv before being written here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir optimum optimum-onnx onnx onnxruntime

# Note: this prints
#     The library name was inferred as `sentence_transformers`, which is not
#     installed. Falling back to `transformers` to avoid breaking the export.
# That is expected and benign FOR THIS MODEL AND TASK. Both paths were compared:
# `--task feature-extraction` exports the transformer body emitting
# last_hidden_state either way, and a parity run against the fallback export
# scored cosine 1.00000000 — identical to the export made with
# sentence_transformers present. sentence_transformers is deliberately NOT
# installed here so the image's model comes from the path that was verified,
# rather than from a different one that merely looks equivalent.
RUN python -m optimum.exporters.onnx \
        --model intfloat/multilingual-e5-small \
        --task feature-extraction \
        /out \
 && test -f /out/model.onnx \
 && test -f /out/tokenizer.json

# Load the graph and run one inference. `test -f` proves a file arrived; this
# proves it is a model. A graph that is present but broken would otherwise
# surface at Lambda INIT — in CloudWatch, minutes later, on a 10-minute loop.
#
# A script rather than `RUN python -c "..."`: a multi-line `python -c` in a
# Dockerfile is a shell-quoting exercise where a mistake costs exactly the loop
# this check exists to avoid, and this version can be run on a laptop against the
# same export.
COPY agenticRAG/verify_onnx_export.py /tmp/verify_onnx_export.py
RUN python /tmp/verify_onnx_export.py /out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime. No torch, no transformers, no optimum.
#
# BASE IMAGE: a plain python image, NOT public.ecr.aws/lambda/python. The Lambda
# base image ships a runtime interface client that polls the Runtime API, and so
# does the Web Adapter — two things competing for the same role. LWA container
# deployments use an ordinary base and let the adapter be the runtime interface.
# The consequence to remember: no AWS base image means no preinstalled boto3, so
# it is declared in requirements-agent-runtime.txt.
# ─────────────────────────────────────────────────────────────────────────────
FROM public.ecr.aws/docker/library/python:3.12-slim

# LWA as an extension. Pinned to 0.9.1 — not "latest", and not 1.1.0 which also
# exists — because 0.9.1 is what LambdaAdapterLayerX86:25 contains, and that
# layer is what the Phase 0 spike actually proved streams correctly behind API
# Gateway. Moving this pin means the streaming evidence no longer applies to
# what is deployed.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 \
     /lambda-adapter /opt/extensions/lambda-adapter

# The exported model, and nothing that made it.
COPY --from=export /out /opt/model

WORKDIR /var/task

COPY agenticRAG/requirements-agent-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-agent-runtime.txt

# LAYOUT MIRRORS THE REPO, deliberately. db/postgres.py, nodes/memory.py,
# mcp/client.py and nodes/_persona_loader.py all resolve configuration as
# `Path(__file__).resolve().parents[3] / "config"`. Put langgraph_agents at the
# root of /var/task — the way the CRUD zip does — and parents[3] becomes /var,
# so config/ is silently not found: yaml lookups return {} and fall back to
# defaults with no error, while persona files simply are not there.
COPY agenticRAG/langgraph_agents ./agenticRAG/langgraph_agents
COPY config ./config
COPY agenticRAG/run.sh ./run.sh
RUN chmod 755 ./run.sh

# Context là gốc repo nên với sang text-to-motion được. Đặt ở /var/task để `import
# vva_motion` giống hệt phía Kimodo.
COPY text-to-motion/kimodo/vva_motion ./vva_motion

# One ENV per concern rather than a single continued instruction, so each can
# carry the comment that explains it. Comments inside a `\`-continued ENV are a
# coin toss across builder versions, and this is not the place to find out —
# every mistake here costs a 10-minute CI round trip.
ENV PYTHONPATH=/var/task/agenticRAG
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# THE line Phase 0 tested. Default is "buffered"; without it API Gateway
# receives one whole body and /chat looks "slightly slow" rather than broken.
ENV AWS_LWA_INVOKE_MODE=response_stream

# On-demand INIT is capped at 10s. Async init lets a slow start spill into the
# first invocation instead of failing the execution environment.
ENV AWS_LWA_ASYNC_INIT=true

# Without this LWA probes "/" and waits — this app has no root route.
ENV AWS_LWA_READINESS_CHECK_PATH=/health

# Embeddings on onnxruntime, reading the model baked in above. Parity with the
# torch path was measured against 200 real KB rows before this default was
# allowed to differ from local: scripts/verify_onnx_parity.py.
ENV EMBEDDING_BACKEND=onnx
ENV E5_ONNX_DIR=/opt/model

# MCP discovery off. It runs in the FastAPI lifespan, i.e. inside the INIT
# budget, and both configured servers are stdio — so it would spawn two Python
# subprocesses that cannot work here: kimodo_server is a mock and
# web_search_server wants SearXNG on localhost. Turning Kimodo on later is this
# flag plus a streamable_http entry, no code change.
ENV ENABLE_MCP=false

# Nothing may reach out to HuggingFace at run time. If an import ever does find
# its way to the hub, it should fail offline rather than quietly download
# hundreds of megabytes during a cold start.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Deliberately NOT set: AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap. That belongs to
# the zip+layer packaging (see infra/infra/crud_api_stack.py). In a container the
# adapter is already in /opt/extensions and starts itself.
#
# Also NOT set here: STM_BACKEND, VVA_PG_DSN_PARAM, COGNITO_*, ALLOWED_ORIGINS.
# Those are deployment configuration and belong to the CDK stack, so the same
# image can be pointed at a different database or user pool without a rebuild.

CMD ["./run.sh"]
