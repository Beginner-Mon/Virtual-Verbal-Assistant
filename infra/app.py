#!/usr/bin/env python3
"""VVA Infrastructure — CDK App Entry Point.

Two tracks live side by side. Only one is synthesised by default.

TRACK 2 — cost-optimised (active)
    VvaCharacterStack ── Lambda (no VPC) → Neon over TLS
    VvaAssetStack     ── private S3 + CloudFront, /characters* → the Lambda
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
from infra.kimodo_ecs_stack import KimodoEcsStack

app = cdk.App()

env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)

# ── VPC ─────────────────────────────────────────────────────────────
# Track 1 needs it; Kimodo ECS uses it today. nat_gateways=0 and no interface
# endpoints, so it costs nothing while it sits there.
vpc_stack = VpcStack(app, "VvaVpcStack", env=env)

# ── Track 2: character catalog + assets ─────────────────────────────
# CharacterStack first — AssetStack attaches its Function URL as an origin.
character_stack = CharacterStack(app, "VvaCharacterStack", env=env)

asset_stack = AssetStack(
    app, "VvaAssetStack",
    characters_fn_url=character_stack.fn_url,
    env=env,
)

# ── Kimodo ECS (GPU MCP Server) ─────────────────────────────────────
kimodo_ecs = KimodoEcsStack(app, "VvaKimodoEcsStack", vpc=vpc_stack.vpc, env=env)

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
