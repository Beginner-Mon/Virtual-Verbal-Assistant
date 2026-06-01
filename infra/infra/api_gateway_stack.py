"""API Gateway Stack — REST API for VVA session CRUD.

Regional REST API with Lambda proxy integrations:
    GET    /sessions                        → vva-list-sessions
    DELETE /sessions/{session_id}           → vva-delete-session
    GET    /sessions/{session_id}/messages  → vva-resume-session

Deploy stage: v1
Throttling: 50 req/s sustained, 100 burst
CORS: all origins (lock down in production)

URL structure changes from original FastAPI:
    - DELETE /sessions/{user_id}/{session_id} → DELETE /sessions/{session_id}?user_id=xxx
      (user_id moved to query param for cleaner REST paths)
    - POST /sessions/{session_id}/resume → GET /sessions/{session_id}/messages
      (idempotent read, cursor-based pagination, no Redis STM populate)
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    aws_apigateway as apigw,
    aws_lambda as lambda_,
)
from constructs import Construct


class ApiGatewayStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        fn_list_sessions: lambda_.Function,
        fn_delete_session: lambda_.Function,
        fn_resume_session: lambda_.Function,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── REST API ────────────────────────────────────────────────────

        self.api = apigw.RestApi(
            self, "VvaApi",
            rest_api_name="vva-sessions-api",
            description="VVA Session CRUD API Gateway",
            deploy_options=apigw.StageOptions(
                stage_name="v1",
                throttling_rate_limit=50,    # Sustained req/s
                throttling_burst_limit=100,  # Burst req/s
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=["GET", "DELETE", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization"],
            ),
        )

        # ── Resource tree ───────────────────────────────────────────────
        #
        # /sessions
        # ├── GET                              → list_sessions
        # └── /{session_id}
        #     ├── DELETE                        → delete_session
        #     └── /messages
        #         └── GET                       → resume_session

        # /sessions
        sessions = self.api.root.add_resource("sessions")
        sessions.add_method(
            "GET",
            apigw.LambdaIntegration(fn_list_sessions, proxy=True),
        )

        # /sessions/{session_id}
        session_by_id = sessions.add_resource("{session_id}")
        session_by_id.add_method(
            "DELETE",
            apigw.LambdaIntegration(fn_delete_session, proxy=True),
        )

        # /sessions/{session_id}/messages
        messages = session_by_id.add_resource("messages")
        messages.add_method(
            "GET",
            apigw.LambdaIntegration(fn_resume_session, proxy=True),
        )

        # ── Outputs ─────────────────────────────────────────────────────

        CfnOutput(
            self, "ApiUrl",
            value=self.api.url,
            description="VVA Sessions API base URL",
        )
