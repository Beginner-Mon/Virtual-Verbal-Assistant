"""REST API Gateway — the single front door for every VVA backend call.

    GET    /v1/characters                        GET    /v1/sessions
    GET    /v1/characters/{slug}                 GET    /v1/sessions/{id}
    GET    /v1/characters/{slug}/avatar-profile  DELETE /v1/sessions/{id}
    GET    /v1/health, /v1/health/db             GET|POST /v1/me/memory
                                                 DELETE /v1/me/memory/{fact_id}

Before this, the frontend had to know four API origins and each service carried
its own CORS and its own auth, with no layer where a shared policy could live and
no way to throttle anything. The gateway is that layer.

**REST API, not HTTP API.** The deciding feature is response streaming, which
REST API gained in November 2025 and HTTP API still lacks. It is what lets /chat
join this API later — via an HTTP proxy integration with
`responseTransferMode: STREAM`, which AWS documents for exactly this case
("server-sent events", "time-to-first-byte for chatbots") and which also lifts
the integration timeout to 15 minutes. Choosing HTTP API today would have been
cheaper by $0.25/month and would have locked /chat out permanently.

**The stage is the version.** `stage_name="v1"` puts /v1 in the URL for free.
Do NOT also add an /api/v1 path prefix: a REST API's `event["path"]` excludes the
stage, so the character handler's router sees "/characters" and works unchanged —
but a real prefix would make it "/api/v1/characters", whose first segment is not
"characters", and every request would 404 with nothing else looking wrong.

**Not fronted by CloudFront.** An earlier design put CloudFront in front of this
API for a single hostname. It bought one environment variable and an edge cache
on a four-row catalog that browsers already cache for five minutes; it cost an
extra hop and three silent failure modes, two of which were security-relevant
(a forgotten OriginRequestPolicy 401s everything; a forgotten CACHING_DISABLED
serves one user's sessions to another, above the layer where RLS can help).
CloudFront keeps the job it is good at: the 9-17 MB .vrm files in S3, which
cannot pass through API Gateway's 10 MB buffered payload limit anyway.
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_apigateway as apigw,
    aws_cognito as cognito,
    aws_lambda as lambda_,
)
from constructs import Construct

from infra.origins import resolve as resolve_origins


# Sized against the account's Lambda concurrency quota, which is TEN — not against
# what feels generous. The first values here were 20/40, and a 120-request burst
# proved them wrong on 20-08:
#
#     Lambda Throttles  61      API Gateway 5XXError  61
#     Lambda Errors      0      API Gateway 4XXError   1
#
# Only one request got a 429 from the gateway. The other sixty exhausted Lambda's
# concurrency instead, and API Gateway reports that as a 5XX. Two things were
# wrong with that:
#
#   - A 500 tells the caller "the server broke". A 429 tells it "slow down, try
#     again" — and only one of those is true.
#   - Those sixty throttled invocations consumed the pool this account SHARES with
#     Amplify's Cognito triggers. That is exactly the failure the gateway exists
#     to prevent: a burst on the catalog stopping people from signing up.
#
# A ceiling above what the platform can serve does not limit anything; it just
# moves the failure somewhere less honest. Five leaves roughly half the pool for
# sign-in, which is the thing being protected.
#
# RAISE THESE once the Lambda quota goes up (it is adjustable; the AWS default is
# 1000, this account sits at 10). At that point `reserved_concurrent_executions`
# on the functions becomes legal too, and it is the better tool — it fences the
# concurrency itself rather than a request rate that stands in for it.
#
# Note this is a stage-wide ceiling, not a per-caller one: API Gateway's stage
# throttle is a single token bucket and does not look at identity. Per-caller
# limits need usage plans (REST API supports them) or a WAF rate rule.
_DEFAULT_RATE_LIMIT = 5
_DEFAULT_BURST_LIMIT = 5


class RestApiStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        crud_fn: lambda_.IFunction,
        characters_fn: lambda_.IFunction,
        cognito_pool_id: str,
        agent_fn: lambda_.IFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        allowed_origins = resolve_origins(self.node)

        # ── The API ─────────────────────────────────────────────────────

        self.api = apigw.RestApi(
            self, "VvaRestApi",
            rest_api_name="vva-api",
            description="Single front door for VVA backend calls",
            # Regional rather than edge-optimized: the users are in one region and
            # the callers that matter are browsers already talking to CloudFront
            # for assets. An edge-optimized endpoint would add a CloudFront
            # distribution we neither configure nor need.
            endpoint_configuration=apigw.EndpointConfiguration(
                types=[apigw.EndpointType.REGIONAL],
            ),
            deploy_options=apigw.StageOptions(
                stage_name="v1",
                throttling_rate_limit=ctx("api_rate_limit") or _DEFAULT_RATE_LIMIT,
                throttling_burst_limit=ctx("api_burst_limit") or _DEFAULT_BURST_LIMIT,
                # X-Ray is REST-API-only, and it is the thing that will answer
                # "was that slow in the gateway, the Lambda, or Neon?" without
                # anyone having to add timing code.
                tracing_enabled=True,
                metrics_enabled=True,
            ),
            # Answers OPTIONS for every resource below. Necessary rather than
            # optional: infra/lambda/characters/handler.py rejects any method
            # other than GET/HEAD with a 405, so a preflight that reached the
            # integration would fail the CORS check.
            #
            # This covers the preflight ONLY. Headers on the actual response come
            # from the function — under a proxy integration that is the function's
            # job, and the console's "Enable CORS" does not apply. See
            # infra/lambda/layer/shared/response.py.
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=allowed_origins,
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization"],
                max_age=Duration.hours(1),
            ),
        )

        # ── Gateway responses: make auth failures legible ───────────────
        #
        # Without these, a rejected token is reported to the user as a CORS
        # error. That is not a metaphor — it happened on 22-08:
        #
        #     Access to XMLHttpRequest at '.../v1/sessions/<id>' ... has been
        #     blocked by CORS policy: No 'Access-Control-Allow-Origin' header
        #     is present on the requested resource.
        #
        # while the actual response was a perfectly ordinary
        # 401 UnauthorizedException. API Gateway builds these responses ITSELF,
        # before any integration runs, so FastAPI's CORS middleware never
        # executes and the header is simply absent. The browser then blames CORS,
        # and whoever is debugging goes looking at origins instead of at tokens.
        #
        # '*' rather than the caller's origin, and for a stated reason: the
        # gateway constructs these before it has looked at the request, so there
        # is no origin to reflect. It is safe for exactly these responses —
        # 4XX/5XX shells carrying no user data — and this API authenticates with
        # an Authorization header rather than cookies, so no credentialed request
        # is widened by it. Same reasoning, same wording, as the Amplify auth API
        # in ECA_UI/frontend/amplify/backend.ts, which solved this first.
        #
        # DEFAULT_4XX and DEFAULT_5XX rather than only UNAUTHORIZED: throttling
        # (429) is built the same way, and the stage throttle here is 5 rps — so
        # the first burst would otherwise also arrive dressed as a CORS failure.
        _cors_headers = {
            "Access-Control-Allow-Origin": "'*'",
            "Access-Control-Allow-Headers": "'Content-Type,Authorization'",
        }
        for name, response_type in (
            ("Unauthorized", apigw.ResponseType.UNAUTHORIZED),
            ("AccessDenied", apigw.ResponseType.ACCESS_DENIED),
            ("Default4xx", apigw.ResponseType.DEFAULT_4_XX),
            ("Default5xx", apigw.ResponseType.DEFAULT_5_XX),
        ):
            self.api.add_gateway_response(
                f"{name}Response",
                type=response_type,
                response_headers=_cors_headers,
            )

        # ── Authorizer ──────────────────────────────────────────────────
        # Built over the SAME pool the CRUD function verifies against — the id is
        # passed in from CrudApiStack rather than read from context twice, because
        # two places reading the same key can still end up trusting two different
        # pools, and the symptom is "a valid token is rejected".
        #
        # This does not replace api/auth.py. The authorizer stops garbage at the
        # edge so the function never starts; the function still needs the token's
        # `sub` to bind app.user_id for row-level security. Two layers, different
        # jobs.
        authorizer = apigw.CognitoUserPoolsAuthorizer(
            self, "CognitoAuthorizer",
            cognito_user_pools=[
                cognito.UserPool.from_user_pool_id(self, "Pool", cognito_pool_id),
            ],
        )
        authed = {
            "authorizer": authorizer,
            "authorization_type": apigw.AuthorizationType.COGNITO,
        }

        crud = apigw.LambdaIntegration(crud_fn)
        characters = apigw.LambdaIntegration(characters_fn)

        # ── /characters — public ────────────────────────────────────────
        # No authorizer, on purpose: the catalog is public data and the app reads
        # it before anyone signs in. Attaching one here would make you log in
        # before you could see which characters exist.
        chars = self.api.root.add_resource("characters")
        chars.add_method("GET", characters)
        chars_slug = chars.add_resource("{slug}")
        chars_slug.add_method("GET", characters)
        chars_slug.add_resource("avatar-profile").add_method("GET", characters)

        # ── /health — public, and it has to be ──────────────────────────
        # The EventBridge warmer in crud_api_stack.py calls /health/db on a
        # schedule and holds no token. An authorizer here would turn the warmer
        # into a scheduled 401.
        health = self.api.root.add_resource("health")
        health.add_method("GET", crud)
        health.add_resource("db").add_method("GET", crud)

        # ── /sessions — authenticated ───────────────────────────────────
        sessions = self.api.root.add_resource("sessions")
        sessions.add_method("GET", crud, **authed)
        sessions_id = sessions.add_resource("{session_id}")
        sessions_id.add_method("GET", crud, **authed)
        sessions_id.add_method("DELETE", crud, **authed)

        # ── /me/memory — authenticated ──────────────────────────────────
        me = self.api.root.add_resource("me")
        memory = me.add_resource("memory")
        memory.add_method("GET", crud, **authed)
        memory.add_method("POST", crud, **authed)
        memory.add_resource("{fact_id}").add_method("DELETE", crud, **authed)

        # ── /chat — authenticated, and STREAMED ─────────────────────────
        #
        # The endpoint this API was chosen for. The module docstring above says
        # REST API was picked over HTTP API because REST gained response
        # streaming in 11/2025 and HTTP API still lacks it; this is that debt
        # being collected.
        #
        # `response_transfer_mode=STREAM` is the whole difference from every
        # other route here. Without it the integration is BUFFERED: API Gateway
        # waits for the complete body, so SSE still "works" but the user sits in
        # silence for twenty seconds and then receives the entire answer at once.
        # It fails by looking slow, which is why the Phase 0 spike measured
        # arrival timestamps rather than watching curl.
        #
        # CDK emits the right integration URI for this — verified in the
        # synthesized template during the spike: it ends in
        # `/response-streaming-invocations`, which is what AWS documents
        # Lambda-proxy streaming requires.
        #
        # Optional so the API can be deployed before the agent exists — see the
        # two-step bootstrap in agent_stack.py. Adding a route later is additive;
        # the reverse (deploying without it once the frontend depends on it)
        # would be an outage, so keep passing agent_fn once it is wired.
        if agent_fn is not None:
            chat = self.api.root.add_resource("chat")
            chat.add_method(
                "POST",
                apigw.LambdaIntegration(
                    agent_fn,
                    response_transfer_mode=apigw.ResponseTransferMode.STREAM,
                    # The default integration timeout is 29s, and a chat turn
                    # runs 10-30s — so the default would cut off exactly the
                    # answers worth waiting for. Raisable only for REGIONAL and
                    # private endpoints, which this API is (see the endpoint
                    # configuration above). Matched to the function's own 120s
                    # ceiling: a gateway that gives up first would abandon a
                    # request Lambda keeps billing for.
                    timeout=Duration.seconds(120),
                ),
                **authed,
            )

            # ── /tts and /billing — the agent's other routes ─────────────
            #
            # BUFFERED, not streamed: both return a small JSON body, and STREAM
            # mode gives up caching, VTL and WAF inspection to buy a lower
            # time-to-first-byte that a 200-byte response does not have.
            #
            # Routed even though neither works in production today — TTS is
            # unhosted so POST /tts answers 503, and billing is sandbox-gated so
            # it does the same. That is the point: a 503 from the real endpoint
            # is a correct answer, and it is what lets the frontend drop
            # VITE_API_BASE_URL now instead of keeping a second origin alive for
            # two features that are switched off. Turning TTS on later becomes
            # one environment variable on the function, with no gateway change
            # and no frontend change.
            agent = apigw.LambdaIntegration(agent_fn)

            tts = self.api.root.add_resource("tts")
            tts.add_method("POST", agent, **authed)
            tts.add_resource("{task_id}").add_resource("result").add_method(
                "GET", agent, **authed,
            )

            billing = self.api.root.add_resource("billing")
            billing.add_resource("{proxy+}").add_method("ANY", agent, **authed)

        CfnOutput(
            self, "RestApiUrl",
            value=self.api.url,
            description="Set as VITE_API_GATEWAY_URL in the frontend build",
        )
