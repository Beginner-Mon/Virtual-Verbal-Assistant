"""Streaming probe stack — Phase 0 spike. ANSWERED 21-08; kept as evidence.

    RESULT: STREAMED. LWA's response_stream prelude satisfies API Gateway's
    Lambda-proxy STREAM mode. No Function URL and no shared secret are needed.

    11 events, arrivals tracking the server's emit stamps at a 0.5s cadence,
    spread 4.953s end to end. A buffered response would have delivered all
    eleven within milliseconds of each other. First byte at 2.812s — a cold
    start for a 5.3 MB zip, worth remembering when sizing the real one.

    Synthesised template carried `"ResponseTransferMode": "STREAM"` and an
    integration URI ending `/response-streaming-invocations`, which is exactly
    what the AWS documentation specifies.

Deployed and destroyed in the same session; `describe-stacks` and `get-function`
both confirm nothing was left behind. The code stays committed because it is the
evidence for a decision the rest of the architecture rests on — anyone who doubts
the result can redeploy and re-measure in about two minutes.

Opt-in, and it has to be: this exists to answer one question and then be
destroyed, so it must not appear in a `cdk synth` that someone runs for another
reason — nor sit forgotten in the account as a public unauthenticated endpoint.

    python infra/spike/build_probe.py
    CDK_ENABLE_SPIKE=1 cdk deploy VvaStreamingProbeStack
    python infra/spike/verify_stream.py <ProbeUrl>
    CDK_ENABLE_SPIKE=1 cdk destroy VvaStreamingProbeStack

THE QUESTION
------------
AWS documents LWA as the supported way to stream from a Python Lambda, and
documents API Gateway's Lambda-proxy STREAM mode as requiring a prelude carrying
status code and headers. Nothing states that LWA's prelude IS that prelude —
every LWA streaming example is written against Function URLs. The /chat design
rests on the two agreeing.

Zip and layer, not a container, deliberately: `AWS_LWA_INVOKE_MODE` is an
environment variable and LWA is the same binary in both packagings, so this
answers the same question without folding in a second unknown.

A SEPARATE API, not a route on VvaRestApiStack: a throwaway must be destroyable
without touching the live front door, and a probe route left behind on the real
API would be an unauthenticated endpoint nobody remembers adding.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_apigateway as apigw,
    aws_lambda as lambda_,
)
from constructs import Construct

_INFRA_ROOT = Path(__file__).resolve().parents[1]
_ZIP_PATH = _INFRA_ROOT / "build" / "streaming_probe.zip"

# Same pinned layer VvaCrudApiStack uses, so the probe tests the adapter this
# project actually deploys rather than whatever floats to latest.
_DEFAULT_LWA_LAYER = "arn:aws:lambda:{region}:753240598075:layer:LambdaAdapterLayerX86:25"


class StreamingProbeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not _ZIP_PATH.exists():
            raise FileNotFoundError(
                f"{_ZIP_PATH} not found. Build it first:\n"
                f"    python infra/spike/build_probe.py"
            )

        ctx = self.node.try_get_context
        lwa_layer_arn = ctx("lwa_layer_arn") or _DEFAULT_LWA_LAYER.format(region=self.region)

        fn = lambda_.Function(
            self, "StreamingProbe",
            function_name="vva-streaming-probe",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="run.sh",
            code=lambda_.Code.from_asset(str(_ZIP_PATH)),
            layers=[
                lambda_.LayerVersion.from_layer_version_arn(
                    self, "LwaLayer", lwa_layer_arn,
                ),
            ],
            environment={
                "AWS_LAMBDA_EXEC_WRAPPER": "/opt/bootstrap",
                "AWS_LWA_READINESS_CHECK_PATH": "/health",
                # THE setting under test. Default is "buffered"; without this the
                # probe would report BUFFERED and the answer would be about the
                # configuration rather than about the prelude.
                "AWS_LWA_INVOKE_MODE": "response_stream",
                "PORT": "8080",
            },
            memory_size=512,
            # The probe emits for ~5s and API Gateway's own ceiling is what
            # matters here, not this one. Kept well clear so a timeout cannot be
            # mistaken for a buffering result.
            timeout=Duration.seconds(60),
            description="THROWAWAY — Phase 0 spike, LWA response_stream behind API Gateway",
        )

        api = apigw.RestApi(
            self, "ProbeApi",
            rest_api_name="vva-streaming-probe",
            endpoint_configuration=apigw.EndpointConfiguration(
                types=[apigw.EndpointType.REGIONAL],
            ),
            deploy_options=apigw.StageOptions(stage_name="v1"),
        )

        integration = apigw.LambdaIntegration(
            fn,
            # The line the whole spike exists to exercise.
            response_transfer_mode=apigw.ResponseTransferMode.STREAM,
            timeout=Duration.seconds(60),
        )

        # No authorizer. The probe carries no data and lives for minutes; adding
        # Cognito would mean minting a token to answer a question about
        # transport encoding. It is destroyed in the same session it is created.
        api.root.add_resource("probe").add_method("GET", integration)
        api.root.add_resource("health").add_method("GET", integration)

        CfnOutput(self, "ProbeUrl", value=f"{api.url}probe")
