"""CRUD API Stack — sessions and user memory on Lambda + Neon.

    GET    /sessions            GET|POST /me/memory
    GET    /sessions/{id}       DELETE   /me/memory/{fact_id}
    DELETE /sessions/{id}       GET      /health, /health/db

Track 2, and shaped like VvaCharacterStack: no VPC configuration, because Neon
is a public TLS endpoint and putting the function in the project's private
isolated subnets would need either a NAT gateway (~$32/month) or interface
endpoints (~$7/month each, per AZ) purely to reach something already on the
internet — while adding ENI attachment to every cold start.

Two things differ from the characters function, and both follow from this data
being per-user rather than a public catalog:

**No CloudFront.** Nothing here is cacheable, so the CDN would earn nothing —
and it would cost something real: with OAC, CloudFront signs requests to a
Function URL WITHOUT hashing the body, so every POST would need the browser to
compute and send `x-amz-content-sha256`. That is a change to every call site in
the frontend, in exchange for a cache that never hits.

**FunctionUrlAuthType.NONE.** Not "unauthenticated" — the application
authenticates. api/auth.py verifies a Cognito ID token on every route and there
is no anonymous path in any environment, so IAM auth on the URL would only add
SigV4 signing that a browser cannot do. Identity is checked where identity is
known, rather than at the edge by a mechanism that cannot see the user.

Prerequisite, created out of band because it is a secret:

    aws ssm put-parameter --name /vva/neon/dsn-pooler --type SecureString \\
        --value 'postgresql://USER:PASS@ep-xxx-pooler.c-NN.us-east-1.aws.neon.tech/neondb?sslmode=require'

Note this one is the POOLED hostname, unlike the characters function. See the
comment on _DEFAULT_DSN_PARAM below.

Build the package before deploying — the asset is a prebuilt zip, not a
directory CDK bundles:

    python infra/build_crud_api.py
    cdk deploy VvaCrudApiStack
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_scheduler as scheduler,
)
from constructs import Construct

_INFRA_ROOT = Path(__file__).resolve().parents[1]
_ZIP_PATH = _INFRA_ROOT / "build" / "crud_api.zip"

# The POOLED Neon endpoint, deliberately — the opposite of every other consumer
# in this repo, which uses the direct one.
#
# The standing rule is "direct, never -pooler", on the grounds that PgBouncer in
# transaction mode breaks the prepared statements asyncpg uses by default. That
# was measured against this database on 18-08 and turned out not to hold: 400
# queries over 5 distinct statements and 10 concurrent workers ran through the
# pooled endpoint with no prepared-statement errors at all. Neon's pooler tracks
# and re-prepares them. So the rule's premise is true of PgBouncer in general and
# false here.
#
# What the pooler buys: a Lambda scales out to N ephemeral containers, each
# opening its own connections, which is exactly the connection churn PgBouncer
# exists to absorb.
#
# What it costs, also measured: session state set with a plain `SET` LEAKS
# between clients. asyncpg resets a connection when returning it to its own pool,
# but through PgBouncer that reset reaches some backend rather than the one
# holding the state — 98 of 200 reads saw another client's value, against 0 on
# the direct endpoint. Anything session-scoped must therefore go through
# PostgresClient.user_scope(), which uses set_config(..., true) and is
# transaction-scoped. See db/postgres.py.
_DEFAULT_DSN_PARAM = "/vva/neon/dsn-pooler"

# AWS Lambda Web Adapter, published by AWS under account 753240598075. Override
# with `-c lwa_layer_arn=...` when a newer version is wanted; the version is
# pinned rather than floating so a redeploy cannot change the runtime underneath
# a working function.
_DEFAULT_LWA_LAYER = (
    "arn:aws:lambda:{region}:753240598075:layer:LambdaAdapterLayerX86:25"
)

_DEFAULT_ORIGINS = [
    "https://release.d32nf9wwqqt016.amplifyapp.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# The production Cognito pool, as defaults rather than required context.
#
# These are PUBLIC identifiers — they ship inside the frontend bundle and ride on
# every Cognito request. Nothing is protected by their obscurity.
#
# They are defaults, not required flags, because this stack stopped being
# optional on 20-08: the REST API references its function, so `app.py` has to
# construct it unconditionally. With the old `-c cognito_user_pool_id=...` gate,
# one `cdk deploy` that forgot the flag would have silently removed routes from a
# live API. Override per deploy when pointing at a different pool.
_DEFAULT_COGNITO_POOL_ID = "us-east-1_6mSqgs4BA"
_DEFAULT_COGNITO_CLIENT_ID = "1rsd1gn5i3heshuo0hf1s6cvm"


class CrudApiStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        dsn_param = ctx("crud_dsn_param") or _DEFAULT_DSN_PARAM
        lwa_layer_arn = ctx("lwa_layer_arn") or _DEFAULT_LWA_LAYER.format(region=self.region)

        allowed_origins = [
            o.strip() for o in (ctx("allowed_origins") or "").split(",") if o.strip()
        ] or _DEFAULT_ORIGINS

        cognito_region = ctx("cognito_region") or self.region
        cognito_pool_id = ctx("cognito_user_pool_id") or _DEFAULT_COGNITO_POOL_ID
        cognito_client_id = ctx("cognito_app_client_id") or _DEFAULT_COGNITO_CLIENT_ID

        # Kept even though the defaults above make it unreachable in practice.
        # api/auth.py has no mode in which missing provider config is survivable —
        # it raises from the app's lifespan — so on Lambda the omission would
        # surface as an INIT failure with a stack trace and no hint that the cause
        # was a deploy-time value. Cheaper to say so at synth.
        if not cognito_pool_id or not cognito_client_id:
            raise ValueError(
                "VvaCrudApiStack needs the Cognito pool it should trust. Pass:\n"
                "  cdk deploy VvaCrudApiStack \\\n"
                "    -c cognito_user_pool_id=us-east-1_xxxxxxxxx \\\n"
                "    -c cognito_app_client_id=xxxxxxxxxxxxxxxxxxxxxx\n"
                "Use the PRODUCTION pool here. A sandbox pool from `ampx sandbox` "
                "is for local development, and the whole point of separating them "
                "is that a token from one is rejected by the other."
            )

        # Exposed so VvaRestApiStack can build a CognitoUserPoolsAuthorizer over
        # the same pool this function verifies tokens against. Two places trusting
        # two different pools is a failure that looks like "valid token rejected".
        self.cognito_pool_id = cognito_pool_id
        self.cognito_client_id = cognito_client_id
        self.cognito_region = cognito_region

        if not _ZIP_PATH.exists():
            raise FileNotFoundError(
                f"{_ZIP_PATH} not found. Build it first:\n"
                f"    python infra/build_crud_api.py\n"
                f"The asset is a prebuilt zip because run.sh has to carry an "
                f"executable bit that a Windows filesystem cannot express."
            )

        # ── Function ────────────────────────────────────────────────────

        self.fn = lambda_.Function(
            self, "CrudApi",
            function_name="vva-crud-api",
            runtime=lambda_.Runtime.PYTHON_3_12,
            # Not a Python handler. Under the Web Adapter the "handler" is the
            # script that starts the HTTP server; /opt/bootstrap comes from the
            # layer and execs it.
            handler="run.sh",
            code=lambda_.Code.from_asset(str(_ZIP_PATH)),
            layers=[
                lambda_.LayerVersion.from_layer_version_arn(
                    self, "LwaLayer", lwa_layer_arn,
                ),
            ],
            environment={
                "AWS_LAMBDA_EXEC_WRAPPER": "/opt/bootstrap",
                # Without this LWA probes "/" and waits, because this app has no
                # root route to answer with.
                "AWS_LWA_READINESS_CHECK_PATH": "/health",
                "PORT": "8080",
                "VVA_PG_DSN_PARAM": dsn_param,
                "AUTH_PROVIDER": "cognito",
                "COGNITO_REGION": cognito_region,
                "COGNITO_USER_POOL_ID": cognito_pool_id,
                "COGNITO_APP_CLIENT_ID": cognito_client_id,
                "ALLOWED_ORIGINS": ",".join(allowed_origins),
                "LOG_LEVEL": "INFO",
                # The embedding model is never loaded on this path, but the
                # environment is set anyway: if an import ever does reach
                # sentence-transformers, it should fail offline rather than
                # quietly download 500 MB during a cold start.
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            # 512 MB, one step above the 256 MB that measurement settled on for
            # vva-characters. Cold start here is CPU-bound in a way that one is
            # not — uvicorn, FastAPI and pydantic all import before the first
            # request — and character_stack.py records the lesson that less
            # memory made cold starts both slower AND more expensive. Worth
            # re-measuring against 256 MB from the INIT_DURATION line.
            memory_size=512,
            # 30s: generous for a single indexed query. The ceiling that matters
            # is a cold start landing on a suspended Neon compute.
            timeout=Duration.seconds(30),
            # No reserved concurrency, and not by preference: AWS refuses any
            # reservation that leaves the account under 100 unreserved units, and
            # this account's whole limit is 10 (measured 19-08). Reserving even 1
            # fails the deploy outright — the earlier `reserved_concurrent_
            # executions=5` here would never have applied.
            #
            # What that costs: those 10 units are shared with the Amplify Cognito
            # triggers (post-confirmation, pre-token-generation), so a burst
            # against this API can throttle sign-in. Nothing in this stack can
            # prevent that; the account limit is the only cap, and it caps the
            # wrong thing — total, not per-function. Owner accepted this on 19-08
            # rather than raise the quota. Revisit when either the quota goes up
            # (then reserve, ~5) or sign-in starts returning throttle errors.
            description="Sessions + user memory, served from Neon",
        )

        # ── IAM: read the DSN SecureString ──────────────────────────────

        self.fn.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter{dsn_param}"],
        ))
        # Scoped by kms:ViaService so this grant cannot be used against anything
        # that is not SSM.
        self.fn.add_to_role_policy(iam.PolicyStatement(
            actions=["kms:Decrypt"],
            resources=[f"arn:aws:kms:{self.region}:{self.account}:key/*"],
            conditions={"StringEquals": {"kms:ViaService": f"ssm.{self.region}.amazonaws.com"}},
        ))

        # ── No Function URL ─────────────────────────────────────────────
        #
        # There was one, with FunctionUrlAuthType.NONE, back when this function
        # was reached directly by the browser. It was removed on 20-08 before the
        # stack was ever deployed, because VvaRestApiStack now fronts this
        # function and a public Function URL would be a second door with nothing
        # standing at it: the gateway's throttle and its Cognito authorizer would
        # both be one URL away from being bypassed.
        #
        # That matters more than it sounds. Application-level auth still applies
        # (api/auth.py verifies the token on every route), so a bypass would not
        # expose data — but throttling is the whole reason the gateway exists,
        # and this account's ten concurrent executions are shared with Amplify's
        # Cognito triggers. A bypassable rate limit protects nothing.
        #
        # The same reasoning removed the characters Function URL in
        # character_stack.py. One way in.

        # ── Warmer ──────────────────────────────────────────────────────

        self._add_warmer(ctx)

    def _add_warmer(self, ctx) -> None:
        """Keep the container AND Neon's compute warm during working hours.

        Two cold starts stack on the first request of the day: the Lambda
        container (~2-4s) and Neon's compute, which suspends when idle and takes
        0.5-2s to resume. This calls /health/db, so it addresses both.

        The schedule is a window, not a heartbeat, and the reason is arithmetic.
        The branch autoscales between 0.25 and 2 CU (confirmed in the console
        19-08). Only the floor matters here: a /health/db ping is one `SELECT 1`
        and never gives the autoscaler a reason to climb, so keeping the database
        awake costs the 0.25 CU minimum per wall-clock hour.

            24/7        730h x 0.25 = 182 CU-hrs   over budget
            13h/day     390h x 0.25 =  98 CU-hrs   no headroom
            10h weekday 220h x 0.25 =  55 CU-hrs   fits, ~45% spare

        The 2 CU ceiling is what makes the ~45% spare thinner than it looks: real
        query load is billed at whatever CU the autoscaler picks, so an hour spent
        at the ceiling costs eight of these. The warmer's share is fixed; what can
        overrun the budget is traffic, and vector search on /chat is the shape of
        query that climbs. Watch the console figure against 55, not against 100.

        A separate tiny function rather than a scheduler target on the API
        itself: EventBridge invokes Lambda with a JSON event, and the Web Adapter
        expects an HTTP request shape. This one synthesises that shape and invokes
        the function directly.

        It used to fetch the Function URL over HTTPS, which exercised the same
        path a browser takes. That URL is gone (see above), and going through the
        REST API instead is not an option: the API stack already depends on this
        one, so pointing the warmer at the API would close a dependency cycle that
        CloudFormation cannot deploy.

        Invoking directly costs one thing worth naming — it no longer proves the
        public path works end to end, only that the function and Neon are warm,
        which is what the warmer is actually for. Use the verification curls for
        the public path.
        """
        warmer = lambda_.Function(
            self, "Warmer",
            function_name="vva-crud-api-warmer",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_inline(
                "import json, os, boto3\n"
                "\n"
                "_lambda = boto3.client('lambda')\n"
                "\n"
                "# Payload format 2.0, which is what the Lambda Web Adapter parses.\n"
                "# rawPath is the route; the rest is the minimum LWA needs to build\n"
                "# an HTTP request out of it.\n"
                "_EVENT = {\n"
                "    'version': '2.0',\n"
                "    'rawPath': '/health/db',\n"
                "    'rawQueryString': '',\n"
                "    'headers': {'host': 'warmer.internal'},\n"
                "    'requestContext': {\n"
                "        'http': {'method': 'GET', 'path': '/health/db',\n"
                "                 'protocol': 'HTTP/1.1', 'sourceIp': '127.0.0.1'},\n"
                "    },\n"
                "    'isBase64Encoded': False,\n"
                "}\n"
                "\n"
                "def handler(event, context):\n"
                "    result = _lambda.invoke(\n"
                "        FunctionName=os.environ['TARGET_FUNCTION'],\n"
                "        InvocationType='RequestResponse',\n"
                "        Payload=json.dumps(_EVENT).encode(),\n"
                "    )\n"
                "    body = json.loads(result['Payload'].read() or b'{}')\n"
                "    return {'statusCode': body.get('statusCode')}\n"
            ),
            environment={"TARGET_FUNCTION": self.fn.function_name},
            memory_size=128,
            timeout=Duration.seconds(25),
            description="Warms vva-crud-api and Neon's compute; invokes /health/db directly",
        )
        self.fn.grant_invoke(warmer)

        schedule_expression = ctx("warmer_schedule") or "cron(*/5 8-17 ? * MON-FRI *)"
        timezone = ctx("warmer_timezone") or "Asia/Ho_Chi_Minh"

        scheduler_role = iam.Role(
            self, "WarmerSchedulerRole",
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"),
        )
        warmer.grant_invoke(scheduler_role)

        scheduler.CfnSchedule(
            self, "WarmerSchedule",
            name="vva-crud-api-warmer",
            flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(
                mode="OFF",
            ),
            schedule_expression=schedule_expression,
            schedule_expression_timezone=timezone,
            target=scheduler.CfnSchedule.TargetProperty(
                arn=warmer.function_arn,
                role_arn=scheduler_role.role_arn,
            ),
            description="Working-hours warm-up; see _add_warmer for the CU-hour maths",
        )
