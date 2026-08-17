"""Character Stack — Track 2 (cost-optimised): the catalog API on Lambda + Neon.

    GET /characters
    GET /characters/{slug}
    GET /characters/{slug}/avatar-profile

The function deliberately has **no VPC configuration**. Neon is a public TLS
endpoint, so putting this Lambda in the project's private isolated subnets
would require either a NAT gateway (~$32/month) or interface VPC endpoints
(~$7/month each, per AZ) purely to reach a service that is already on the
internet — and would add ENI attachment to every cold start. Outside the VPC
it is both cheaper and faster. That inversion is the whole point of Track 2;
Track 1 (lambda_stack.py + database_stack.py) keeps the in-VPC/RDS design and
stays undeployed.

Exposed through a Lambda Function URL rather than API Gateway: CloudFront sits
in front and caches the catalog at the edge, so the origin serves roughly one
request per TTL and API Gateway's per-request features earn nothing here.
AWS_IAM auth on the URL means only CloudFront's OAC can invoke it.

Prerequisite, created out of band because it is a secret and must not live in
a CloudFormation template:

    aws ssm put-parameter --name /vva/neon/dsn --type SecureString \\
        --value 'postgresql://USER:PASS@ep-xxx.c-NN.us-east-1.aws.neon.tech/neondb?sslmode=require'

Use the DIRECT endpoint: the pooled hostname with only `-pooler` removed. Newer
Neon hostnames carry a `.c-NN` segment that must be KEPT — dropping it as well
fails authentication.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import jsii
from aws_cdk import (
    CfnOutput,
    Duration,
    ILocalBundling,
    Stack,
    BundlingOptions,
    aws_iam as iam,
    aws_lambda as lambda_,
)
from constructs import Construct

_INFRA_ROOT = Path(__file__).resolve().parents[1]
_LAMBDA_DIR = _INFRA_ROOT / "lambda"

# Default; override at deploy time with `-c neon_dsn_param=/some/other/path`.
_DEFAULT_DSN_PARAM = "/vva/neon/dsn"

# Extensions that make a build host-specific. Their presence means the artifact
# is only valid for the OS that produced it.
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")


@jsii.implements(ILocalBundling)
class _LocalPurePythonBundling:
    """Build the layer on the host instead of inside the Lambda Docker image.

    CDK bundles layers in a container because packages with C extensions ship
    compiled binaries — a wheel built on Windows (`.pyd`) will not load on
    Amazon Linux, which wants `.so`. That protection is real, and irrelevant
    here: this layer is pg8000 and its dependencies, all of which publish
    `py3-none-any` wheels. Verified against the previously Docker-built asset in
    cdk.out — 56 `.py` files, zero binaries. A Linux container would spend its
    time producing exactly what the host already produces.

    The guard is the point of this class, not the speed. After installing we
    scan the output for native extensions; if any appear, the build is
    host-specific, we discard it and return False so CDK falls back to Docker.
    Without that check, adding a compiled dependency one day would yield a layer
    that imports cleanly on the developer's machine and ImportErrors in
    production — a failure that only shows up after deploy.

    (If a compiled dependency ever becomes permanent, the other way out is
    `pip install --platform manylinux2014_x86_64 --only-binary=:all:`, which
    fetches Linux wheels from a Windows host. Sharper edges, so not used while
    everything is pure Python.)
    """

    def try_bundle(self, output_dir: str, *_args, **_kwargs) -> bool:
        out = Path(output_dir)
        python_dir = out / "python"
        requirements = _LAMBDA_DIR / "layer" / "requirements.txt"
        shared_src = _LAMBDA_DIR / "layer" / "shared"

        try:
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "-r", str(requirements),
                    "--target", str(python_dir),
                    "--no-cache-dir", "--quiet", "--disable-pip-version-check",
                ],
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"[layer] local bundling failed ({exc}); falling back to Docker")
            shutil.rmtree(out, ignore_errors=True)
            return False

        native = [
            p for p in python_dir.rglob("*")
            if p.suffix.lower() in _NATIVE_SUFFIXES
        ]
        if native:
            names = ", ".join(sorted({p.name for p in native})[:5])
            print(
                f"[layer] found {len(native)} native extension(s) ({names}) — this "
                f"layer is no longer pure Python and must be built for Amazon "
                f"Linux. Falling back to Docker."
            )
            shutil.rmtree(out, ignore_errors=True)
            return False

        shutil.copytree(shared_src, python_dir / "shared", dirs_exist_ok=True)
        print(f"[layer] bundled locally: {len(list(python_dir.rglob('*.py')))} .py files, 0 binaries")
        return True


class CharacterStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        dsn_param = self.node.try_get_context("neon_dsn_param") or _DEFAULT_DSN_PARAM

        # ── Layer ───────────────────────────────────────────────────────
        # Same source as Track 1's vva-shared-layer. It is built again here
        # rather than shared because the two tracks deploy independently and
        # Track 1 is never deployed — importing its layer would make this
        # stack depend on a stack that must not exist.

        shared_layer = lambda_.LayerVersion(
            self, "VvaSharedLayerTrack2",
            layer_version_name="vva-shared-layer-track2",
            code=lambda_.Code.from_asset(
                str(_LAMBDA_DIR / "layer"),
                bundling=BundlingOptions(
                    # Tried first; falls back to `image` below when it returns
                    # False (native extensions present, or pip unavailable).
                    local=_LocalPurePythonBundling(),
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

        # ── Function ────────────────────────────────────────────────────

        self.fn_characters = lambda_.Function(
            self, "Characters",
            function_name="vva-characters",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(_LAMBDA_DIR / "characters")),
            layers=[shared_layer],
            environment={
                "DB_MODE": "neon",
                "NEON_DSN_PARAM": dsn_param,
            },
            # 10s: one indexed read of a four-row table. The only slow path is a
            # cold start landing on a Neon compute that has scaled to zero.
            timeout=Duration.seconds(10),
            # 256 MB, and measured rather than guessed. 128 MB was tried on the
            # theory that a function waiting on a network round trip does not
            # need CPU. True while warm — 5 ms became 8 ms — but cold start is
            # CPU-bound (module imports, pg8000, the TLS handshake) and went from
            # 1260 ms to 3252 ms.
            #
            # That also makes it more expensive, not less: a cold invocation
            # costs 0.125 GB x 3.252 s = 0.41 GB-s at 128 MB against
            # 0.25 GB x 1.26 s = 0.32 GB-s at 256 MB. And cold is the common case
            # here precisely because CloudFront caches the catalog so well — the
            # origin is hit rarely enough that containers rarely stay warm.
            #
            # Peak usage is ~103 MB, so 128 MB would also leave only 25 MB of
            # headroom.
            memory_size=256,
            description="GET /characters — VRM catalog served from Neon",
        )

        # ── IAM: read the DSN SecureString ──────────────────────────────

        self.fn_characters.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter{dsn_param}"
            ],
        ))
        # SecureString decryption via the AWS-managed SSM key. Scoped by
        # kms:ViaService so this grant cannot be used against anything else.
        self.fn_characters.add_to_role_policy(iam.PolicyStatement(
            actions=["kms:Decrypt"],
            resources=[f"arn:aws:kms:{self.region}:{self.account}:key/*"],
            conditions={
                "StringEquals": {
                    "kms:ViaService": f"ssm.{self.region}.amazonaws.com"
                }
            },
        ))

        # ── Function URL ────────────────────────────────────────────────
        # AWS_IAM, so the URL is not publicly callable. AssetStack attaches it
        # as a CloudFront origin with OAC, which signs requests with SigV4.

        self.fn_url = self.fn_characters.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.AWS_IAM,
        )

        CfnOutput(
            self, "CharactersFunctionUrl",
            value=self.fn_url.url,
            description="Direct Function URL (IAM-signed only; browsers use CloudFront)",
        )
