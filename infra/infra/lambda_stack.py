"""Lambda Stack - CRUD session endpoints for VVA.

Three Lambda functions for session management:
    vva-list-sessions    - GET /sessions
    vva-delete-session   - DELETE /sessions/{session_id}
    vva-resume-session   - GET /sessions/{session_id}/messages

All functions:
    - Run in VPC private subnets (same as RDS)
    - Connect to RDS via RDS Proxy (IAM Auth + TLS)
    - Read connection config from SSM Parameter Store
    - Dependencies (pg8000) packaged in shared Lambda Layer
    - Runtime: Python 3.12, 256MB, 30s timeout

Layer bundling requires Docker (CDK uses the Python 3.12 Lambda image
to pip install pg8000 into the layer).
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    Duration,
    Stack,
    BundlingOptions,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_rds as rds,
)
from constructs import Construct

# Resolve paths relative to this file (infra/infra/lambda_stack.py)
# → infra/ is the project root → lambda/ is at infra/lambda/
_INFRA_ROOT = Path(__file__).resolve().parents[1]
_LAMBDA_DIR = _INFRA_ROOT / "lambda"


class LambdaStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.Vpc,
        rds_proxy: rds.DatabaseProxy,
        db_instance: rds.DatabaseInstance,
        db_param_prefix: str,
        sg_rds_proxy: ec2.SecurityGroup,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Security Group ──────────────────────────────────────────────

        self.sg_lambda = ec2.SecurityGroup(
            self, "SgLambda",
            vpc=vpc,
            security_group_name="vva-sg-lambda",
            description="Lambda functions - outbound to RDS Proxy and VPC Endpoints",
            allow_all_outbound=False,
        )

        # Lambda → RDS Proxy (PostgreSQL)
        self.sg_lambda.add_egress_rule(
            peer=sg_rds_proxy,
            connection=ec2.Port.tcp(5432),
            description="PostgreSQL to RDS Proxy",
        )
        ec2.CfnSecurityGroupIngress(
            self, "SgRdsProxyIngressFromLambda",
            group_id=sg_rds_proxy.security_group_id,
            ip_protocol="tcp",
            from_port=5432,
            to_port=5432,
            source_security_group_id=self.sg_lambda.security_group_id,
            description="PostgreSQL from Lambda",
        )

        # Lambda → VPC Interface Endpoints (HTTPS 443)
        # Needed for SSM, STS + CloudWatch Logs in isolated subnets
        self.sg_lambda.add_egress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(443),
            description="HTTPS to VPC Endpoints (SSM, STS, CloudWatch)",
        )

        # ── Lambda Layer (pg8000 + shared utilities) ────────────────────
        # Docker bundling: CDK pulls the Python 3.12 Lambda image, runs
        # pip install, and copies shared/ into the layer's python/ dir.

        shared_layer = lambda_.LayerVersion(
            self, "VvaSharedLayer",
            layer_version_name="vva-shared-layer",
            code=lambda_.Code.from_asset(
                str(_LAMBDA_DIR / "layer"),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash", "-c",
                        " && ".join([
                            "pip install -r /asset-input/requirements.txt -t /asset-output/python --no-cache-dir --quiet",
                            "cp -r /asset-input/shared /asset-output/python/shared",
                        ]),
                    ],
                ),
            ),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="VVA shared deps (pg8000) and utilities (db, response)",
        )

        # ── Shared config for all Lambda functions ──────────────────────

        env_vars = {
            "DB_PARAM_PREFIX": db_param_prefix,
        }

        common_props = dict(
            runtime=lambda_.Runtime.PYTHON_3_12,
            layers=[shared_layer],
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_group_name="Private"),
            security_groups=[self.sg_lambda],
            environment=env_vars,
            timeout=Duration.seconds(30),
            memory_size=256,
        )

        # ── Lambda Functions ────────────────────────────────────────────

        self.fn_list_sessions = lambda_.Function(
            self, "ListSessions",
            function_name="vva-list-sessions",
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(_LAMBDA_DIR / "list_sessions")),
            description="GET /sessions - list user sessions",
            **common_props,
        )

        self.fn_delete_session = lambda_.Function(
            self, "DeleteSession",
            function_name="vva-delete-session",
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(_LAMBDA_DIR / "delete_session")),
            description="DELETE /sessions/{session_id} - delete a session",
            **common_props,
        )

        self.fn_resume_session = lambda_.Function(
            self, "ResumeSession",
            function_name="vva-resume-session",
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(_LAMBDA_DIR / "resume_session")),
            description="GET /sessions/{session_id}/messages - cursor pagination",
            **common_props,
        )

        # ── IAM: Lambda → RDS Proxy (IAM Auth) ─────────────────────────
        # grant_connect() adds rds-db:connect permission on this proxy's
        # resource ARN, scoped to the vva_admin database user.

        rds_proxy.grant_connect(self.fn_list_sessions, "vva_admin")
        rds_proxy.grant_connect(self.fn_delete_session, "vva_admin")
        rds_proxy.grant_connect(self.fn_resume_session, "vva_admin")

        # ── IAM: Lambda → SSM Parameter Store (read DB config) ─────────

        ssm_param_arn = f"arn:aws:ssm:{self.region}:{self.account}:parameter{db_param_prefix}/*"

        for fn in [self.fn_list_sessions, self.fn_delete_session, self.fn_resume_session]:
            fn.add_to_role_policy(iam.PolicyStatement(
                actions=["ssm:GetParametersByPath", "ssm:GetParameter"],
                resources=[ssm_param_arn],
            ))
