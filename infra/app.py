#!/usr/bin/env python3
"""VVA Infrastructure — CDK App Entry Point.

Stack dependency chain:
    VpcStack → DatabaseStack → LambdaStack → ApiGatewayStack

Deploy all:  cdk deploy --all
Deploy one:  cdk deploy VvaVpcStack
Diff:        cdk diff
Synth:       cdk synth
"""

import os

import aws_cdk as cdk

from infra.vpc_stack import VpcStack
from infra.database_stack import DatabaseStack
from infra.lambda_stack import LambdaStack
from infra.api_gateway_stack import ApiGatewayStack

app = cdk.App()

# Uncomment and set your AWS Account + Region for region-specific resources
# (VPC endpoints, RDS engine versions, etc.)
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)

# ── 1. VPC ──────────────────────────────────────────────────────────
vpc_stack = VpcStack(app, "VvaVpcStack", env=env)

# ── 2. Database (RDS + Secrets Manager + RDS Proxy) ─────────────────
db_stack = DatabaseStack(
    app, "VvaDbStack",
    vpc=vpc_stack.vpc,
    env=env,
)

# ── 3. Lambda Functions (CRUD endpoints) ────────────────────────────
lambda_stack = LambdaStack(
    app, "VvaLambdaStack",
    vpc=vpc_stack.vpc,
    rds_proxy=db_stack.rds_proxy,
    db_instance=db_stack.db_instance,
    db_param_prefix=db_stack.db_param_prefix,
    sg_rds_proxy=db_stack.sg_rds_proxy,
    env=env,
)

# ── 4. API Gateway ──────────────────────────────────────────────────
api_stack = ApiGatewayStack(
    app, "VvaApiStack",
    fn_list_sessions=lambda_stack.fn_list_sessions,
    fn_delete_session=lambda_stack.fn_delete_session,
    fn_resume_session=lambda_stack.fn_resume_session,
    env=env,
)

app.synth()
