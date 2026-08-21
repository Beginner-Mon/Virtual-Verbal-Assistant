#!/usr/bin/env python3
"""VVA Infrastructure — CDK App Entry Point.

Two tracks live side by side. Only one is synthesised by default.

TRACK 2 — cost-optimised (active)
    VvaCharacterStack ── Lambda (no VPC) → Neon over TLS
    VvaCrudApiStack   ── sessions + user memory, Lambda Web Adapter → Neon (pooled)
    VvaRestApiStack   ── REST API in front of both; the one front door for the API
    VvaAssetStack     ── private S3 + CloudFront, .vrm files ONLY
    VvaVpcStack       ── deployed, ~$0 (no NAT, no interface endpoints)

TRACK 1 — production reference (frozen, NOT synthesised)
    VvaDbStack        ── RDS + RDS Proxy
    VvaLambdaStack    ── session CRUD Lambdas inside the VPC, IAM auth
    VvaApiStack       ── REST API in front of them

    The app moved to Neon on 31/07 and these stacks have never been deployed.
    They are kept as the target architecture for when there is budget for it.
    Synthesising them by default would make `cdk deploy --all` stand up an RDS
    instance nobody uses, so they are opt-in:

        CDK_ENABLE_TRACK1=1 cdk synth

Deploy (Track 2):
    cdk deploy VvaCharacterStack
    cdk deploy VvaAssetStack
Deploy GPU:
    cdk deploy VvaKimodoEcsStack

Name a stack explicitly. `cdk deploy --all` is not the way to use this app.
"""

import os

import aws_cdk as cdk

from infra.vpc_stack import VpcStack
from infra.asset_stack import AssetStack
from infra.character_stack import CharacterStack
from infra.crud_api_stack import CrudApiStack
from infra.agent_stack import AgentStack
from infra.rest_api_stack import RestApiStack
from infra.kimodo_ecs_stack import KimodoEcsStack

app = cdk.App()

# Applied to every taggable resource in every stack below, so cost reports and
# resource searches can separate this project from anything else in the account.
# Set on the App rather than per-stack: a stack added later inherits it without
# anyone having to remember.
cdk.Tags.of(app).add("Project", "ECA")

env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)

# ── VPC ─────────────────────────────────────────────────────────────
# Track 1 needs it; Kimodo ECS uses it today. nat_gateways=0 and no interface
# endpoints, so it costs nothing while it sits there.
vpc_stack = VpcStack(app, "VvaVpcStack", env=env)

# ── Track 2: character catalog + assets ─────────────────────────────
character_stack = CharacterStack(app, "VvaCharacterStack", env=env)

# Independent of CharacterStack since 20-08: the catalog moved to the REST API,
# so this distribution serves only the .vrm files from S3 and needs no Lambda.
asset_stack = AssetStack(app, "VvaAssetStack", env=env)

# ── Track 2: session + user-memory CRUD ─────────────────────────────
# Serves api/crud_app.py under the Lambda Web Adapter.
#
# Unconditional since 20-08. It used to be gated on `-c cognito_user_pool_id`, but
# VvaRestApiStack references this function, so a synth that skipped it would break
# the API — and worse, a `cdk deploy` that forgot the flag would have silently
# removed routes from a live gateway. The pool id now has a default in the stack
# (it is a public value, already in the frontend bundle).
#
# Consequence: every synth now needs the package built.
#     python infra/build_crud_api.py
crud_api_stack = CrudApiStack(app, "VvaCrudApiStack", env=env)

# ── Track 2: the LangGraph agent (container) ────────────────────────
# Serves api/main.py — the graph and /chat. Built by CI and pulled from ECR, so
# no Docker is needed on the machine running `cdk deploy`.
#
#     cdk deploy VvaAgentStack -c agent_bootstrap=1        # repository, once
#     cdk deploy VvaAgentStack -c agent_image_tag=<sha>    # every time after
#
# During bootstrap `fn` is None and VvaRestApiStack simply omits /chat. Adding
# the route later is additive; that is why it is optional rather than required.
agent_stack = AgentStack(app, "VvaAgentStack", env=env)

# ── Track 2: the API gateway ────────────────────────────────────────
# One front door for every backend call. See rest_api_stack.py for why REST API
# rather than HTTP API, and why CloudFront does not sit in front of it.
rest_api_stack = RestApiStack(
    app, "VvaRestApiStack",
    crud_fn=crud_api_stack.fn,
    characters_fn=character_stack.fn_characters,
    cognito_pool_id=crud_api_stack.cognito_pool_id,
    agent_fn=agent_stack.fn,
    env=env,
)

# ── Kimodo ECS (GPU MCP Server) ─────────────────────────────────────
kimodo_ecs = KimodoEcsStack(app, "VvaKimodoEcsStack", vpc=vpc_stack.vpc, env=env)

# ── Phase 0 spike: THROWAWAY (opt-in) ───────────────────────────────
# Answers one question — does LWA's response_stream prelude satisfy API
# Gateway's Lambda-proxy STREAM mode? — and is then destroyed. Gated so it
# cannot appear in a synth someone ran for another reason, and so that
# forgetting to destroy it does not quietly leave a public unauthenticated
# endpoint in the account.
#
#     python infra/spike/build_probe.py
#     CDK_ENABLE_SPIKE=1 cdk deploy  VvaStreamingProbeStack
#     CDK_ENABLE_SPIKE=1 cdk destroy VvaStreamingProbeStack
#
# Delete this block and infra/spike/ once Phase 0 is recorded in the worklog.
if os.getenv("CDK_ENABLE_SPIKE") == "1":
    from infra.streaming_probe_stack import StreamingProbeStack

    StreamingProbeStack(app, "VvaStreamingProbeStack", env=env)

# ── Track 1: production reference (opt-in) ──────────────────────────
if os.getenv("CDK_ENABLE_TRACK1") == "1":
    from infra.database_stack import DatabaseStack
    from infra.lambda_stack import LambdaStack
    from infra.api_gateway_stack import ApiGatewayStack

    db_stack = DatabaseStack(app, "VvaDbStack", vpc=vpc_stack.vpc, env=env)

    lambda_stack = LambdaStack(
        app, "VvaLambdaStack",
        vpc=vpc_stack.vpc,
        rds_proxy=db_stack.rds_proxy,
        db_instance=db_stack.db_instance,
        db_param_prefix=db_stack.db_param_prefix,
        sg_rds_proxy=db_stack.sg_rds_proxy,
        env=env,
    )

    api_stack = ApiGatewayStack(
        app, "VvaApiStack",
        fn_list_sessions=lambda_stack.fn_list_sessions,
        fn_delete_session=lambda_stack.fn_delete_session,
        fn_resume_session=lambda_stack.fn_resume_session,
        env=env,
    )

app.synth()
