"""Agent Stack — the LangGraph service and /chat, as a Lambda container image.

    ECR repo `vva-agent`  ←── CI pushes  vva-agent:<git-sha>
              │
              └── Lambda `vva-agent` (1024 MB, 120s)
                       ├─ Neon (pooled DSN from SSM)
                       ├─ DeepSeek / Gemini (keys from SSM)
                       └─ no VPC, no NAT

Shaped like VvaCrudApiStack — no VPC configuration, because Neon is a public TLS
endpoint and putting the function in the project's private isolated subnets would
need either a NAT gateway (~$32/month) or interface endpoints purely to reach
something already on the internet, while adding ENI attachment to every cold
start.

TWO-STEP BOOTSTRAP, and the reason it is not one step
------------------------------------------------------
A container function cannot be created before an image exists, and the image
cannot be pushed before the repository exists. So:

    # 1. repository only, once
    cdk deploy VvaAgentStack -c agent_bootstrap=1

    # 2. CI builds and pushes vva-agent:<sha>  (deploy-agent.yml)

    # 3. the function, pinned to a tag that now exists
    cdk deploy VvaAgentStack -c agent_image_tag=<sha>

Passing NEITHER raises at synth rather than deploying something. That is
deliberate and copied from crud_api_stack.py's Cognito check: a `cdk deploy` that
silently omitted the function would DELETE a live one, and CloudFormation would
report it as a successful deployment.

IMAGE TAGS ARE IMMUTABLE, never `latest`. `kimodo:latest` in
deploy-production.yml is the counter-example: with a floating tag you cannot tell
which build is running and cannot roll back to a previous one.

NO Function URL. VvaRestApiStack is the one front door; it holds the Cognito
authorizer and the stage throttle, and a second entrance would be one URL away
from bypassing both. Response streaming reaches this function through API
Gateway's AWS_PROXY integration with ResponseTransferMode.STREAM — proved end to
end by the Phase 0 spike, see infra/infra/streaming_probe_stack.py.
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_lambda as lambda_,
)
from constructs import Construct

from infra.origins import resolve as resolve_origins

_REPOSITORY_NAME = "vva-agent"

# The POOLED Neon endpoint, same as the CRUD function and for the same measured
# reason — see the long note in crud_api_stack.py. A Lambda scales out to N
# ephemeral containers each opening its own connections, which is exactly the
# churn PgBouncer exists to absorb; and anything session-scoped goes through
# PostgresClient.user_scope(), which is transaction-scoped and survives it.
_DEFAULT_DSN_PARAM = "/vva/neon/dsn-pooler"

# LLM credentials as SSM SecureStrings rather than environment variables.
# Lambda environment variables are plaintext in the CloudFormation template, and
# CloudFormation's {{resolve:ssm-secure}} dynamic reference is not supported for
# them. llm.py reads these at run time; see _secret_from_ssm there.
_DEFAULT_DEEPSEEK_PARAM = "/vva/llm/deepseek-api-key"
_DEFAULT_GEMINI_PARAM = "/vva/llm/gemini-api-keys"

# Same production pool the CRUD function verifies against. Public identifiers —
# they ship inside the frontend bundle — so they are defaults rather than
# required flags.
_DEFAULT_COGNITO_POOL_ID = "us-east-1_6mSqgs4BA"
_DEFAULT_COGNITO_CLIENT_ID = "1rsd1gn5i3heshuo0hf1s6cvm"


class AgentStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        image_tag = ctx("agent_image_tag")
        bootstrap = str(ctx("agent_bootstrap") or "").strip() in ("1", "true", "yes")

        # ── ECR repository ──────────────────────────────────────────────

        self.repository = ecr.Repository(
            self, "AgentRepository",
            repository_name=_REPOSITORY_NAME,
            # Immutable: pushing over an existing tag is how "which build is
            # running?" becomes unanswerable and rollback becomes impossible.
            # It also makes a CI re-run of the same commit fail loudly rather
            # than silently replace what is deployed.
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep the last 10 images; each is ~1.2 GB",
                    max_image_count=10,
                ),
            ],
            # RETAIN: destroying the stack must not delete the images the
            # running function is built from, nor the history to roll back to.
            removal_policy=RemovalPolicy.RETAIN,
        )

        CfnOutput(
            self, "AgentRepositoryUri",
            value=self.repository.repository_uri,
            description="Push vva-agent:<git-sha> here (deploy-agent.yml)",
        )

        if bootstrap:
            # Step 1 of the bootstrap. The function is deliberately absent
            # because no image exists yet.
            CfnOutput(
                self, "BootstrapNote",
                value="Repository only. Push an image, then deploy with "
                      "-c agent_image_tag=<sha>",
            )
            self.fn = None
            return

        if not image_tag:
            raise ValueError(
                "VvaAgentStack needs the image tag to deploy.\n\n"
                "  First time (repository only):\n"
                "    cdk deploy VvaAgentStack -c agent_bootstrap=1\n\n"
                "  Every time after that:\n"
                "    cdk deploy VvaAgentStack -c agent_image_tag=<git-sha>\n\n"
                "Neither flag is NOT treated as 'skip the function'. A deploy "
                "that quietly omitted it would DELETE the live one, and "
                "CloudFormation would call that a success — the same failure "
                "crud_api_stack.py guards against with its Cognito check."
            )

        # ── Function ────────────────────────────────────────────────────

        allowed_origins = resolve_origins(self.node)
        dsn_param = ctx("crud_dsn_param") or _DEFAULT_DSN_PARAM
        deepseek_param = ctx("deepseek_param") or _DEFAULT_DEEPSEEK_PARAM
        gemini_param = ctx("gemini_param") or _DEFAULT_GEMINI_PARAM

        self.cognito_pool_id = ctx("cognito_user_pool_id") or _DEFAULT_COGNITO_POOL_ID
        cognito_client_id = ctx("cognito_app_client_id") or _DEFAULT_COGNITO_CLIENT_ID
        cognito_region = ctx("cognito_region") or self.region

        self.fn = lambda_.DockerImageFunction(
            self, "Agent",
            function_name="vva-agent",
            # from_ecr, NOT from_image_asset. from_image_asset builds the image
            # during `cdk synth`, on the machine running the deploy — which
            # requires a local Docker daemon. Owner chose CI-only builds on
            # 21-08, so the image arrives already built and CDK only references
            # it. This is also the boundary the plan draws: CDK owns the shape of
            # the infrastructure, CI owns the code.
            code=lambda_.DockerImageCode.from_ecr(
                self.repository, tag_or_digest=image_tag,
            ),
            environment={
                "VVA_PG_DSN_PARAM": dsn_param,
                "DEEPSEEK_API_KEY_PARAM": deepseek_param,
                "GEMINI_API_KEYS_PARAM": gemini_param,
                "AUTH_PROVIDER": "cognito",
                "COGNITO_REGION": cognito_region,
                "COGNITO_USER_POOL_ID": self.cognito_pool_id,
                "COGNITO_APP_CLIENT_ID": cognito_client_id,
                "ALLOWED_ORIGINS": ",".join(allowed_origins),
                "LOG_LEVEL": "INFO",
                # No cache. Short-term memory is a cache over PostgreSQL, and
                # the plan is to run without one for a month and measure the
                # Neon CU-hours it would have saved before paying for anything.
                # Switching later is this value plus STM_TABLE — the code and
                # its tests are already in place. See shared/stm.py.
                "STM_BACKEND": ctx("stm_backend") or "none",
                # ENABLE_MCP, EMBEDDING_BACKEND, E5_ONNX_DIR and the LWA
                # settings are baked into the image: they describe what the
                # image IS, not where it is deployed. See agenticRAG/Dockerfile.
            },
            # 1024 MB, and lower than it looks like it should be. CPU scales
            # with memory (1 vCPU at 1769 MB), so this buys ~0.58 vCPU and makes
            # embedding roughly twice as slow — ~200 ms instead of ~100 ms. That
            # is the right trade because the turn is ~20 SECONDS of waiting on
            # DeepSeek, and Lambda bills memory x wall-clock: halving the memory
            # halves the cost of every second spent waiting. The docs' advice
            # that "over-provisioning memory often lowers cost" is true of
            # CPU-bound work and false of sitting on a socket.
            # Re-measure from the INIT_DURATION line before changing it.
            memory_size=1024,
            # 120s is a COST CEILING, not a safety net. AWS documents that a
            # streaming invocation is billed for its full duration and is NOT
            # stopped when the client disconnects — so main.py's
            # request.is_disconnected() cannot save money here. A real turn is
            # 10-30s; 120s is four times the bad case, and a hung DeepSeek costs
            # 120s x 1 GB rather than the 300s an intuitive value would.
            timeout=Duration.seconds(120),
            # No reserved concurrency: AWS refuses any reservation that leaves
            # the account under 100 unreserved units, and this account's whole
            # limit is 10. Set one and the deploy fails outright. That limit is
            # shared with Amplify's Cognito triggers, and /chat holds a container
            # for 10-30 SECONDS where the CRUD function holds one for ~100 ms —
            # so a few concurrent chats can throttle sign-in. Raise the quota
            # (the AWS default is 1000) before this sees real traffic, then
            # reserve here.
            description="LangGraph agent — /chat, SSE over API Gateway response streaming",
        )

        # ── IAM: read the DSN and the LLM keys ──────────────────────────

        self.fn.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter{param}"
                for param in (dsn_param, deepseek_param, gemini_param)
            ],
        ))
        # Scoped by kms:ViaService so this grant cannot be turned on anything
        # that is not SSM.
        self.fn.add_to_role_policy(iam.PolicyStatement(
            actions=["kms:Decrypt"],
            resources=[f"arn:aws:kms:{self.region}:{self.account}:key/*"],
            conditions={"StringEquals": {"kms:ViaService": f"ssm.{self.region}.amazonaws.com"}},
        ))

        CfnOutput(self, "AgentFunctionName", value=self.fn.function_name)
        CfnOutput(self, "AgentImageTag", value=image_tag)
