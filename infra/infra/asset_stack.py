"""Asset Stack — Track 2: private S3 + CloudFront for the VRM model files.

    https://<dist>/models/{slug}/{hash}.vrm  → S3

ONE job since 20-08: serve the four .vrm files. The character catalog used to
live here too, on a /characters* behavior pointing at a Lambda Function URL; it
moved to VvaRestApiStack, where the rest of the API is.

That leaves this distribution as a plain CDN in front of a private bucket, which
is what it is good at — and what nothing else can do here, since the files are
9-17 MB and API Gateway's buffered payload limit is 10 MB.

The four .vrm files are ~57 MB and were being bundled into every Vite build,
then copied again into dist/, the Android assets directory and the iOS public
directory. Serving them from a CDN takes them out of the bundle entirely.

Two things worth knowing before editing:

1. **CORS belongs on the CloudFront response headers policy, not on the bucket.**
   The browser only ever talks to CloudFront; S3's own CORS configuration is
   never seen. Setting it there and nowhere else is a common way to spend an
   afternoon on a blank screen.

2. **Object keys embed a content hash** (characters/{slug}/{sha256[:8]}.vrm),
   and the resulting URL is stored in characters.vrm_url. Replacing a model
   changes the key, changes the URL, and needs no invalidation — so the cache
   policy can be maximally aggressive without any staleness risk.

The bucket blocks all public access; CloudFront reaches it through Origin
Access Control. Requesting the S3 URL directly returns 403 by design.

MOTION FILES (Task 7) are the first user-derived artifacts on this bucket —
everything above is a static app asset uploaded by a script, not produced at
request time by a stranger's job. That distinction is why they get two things
the VRM files do not:

1. **A lifecycle rule expiring `motions/` after 1 day.** The GPU worker writes
   `motions/<job_id>.bvh` / `.npz` per job; nothing deletes them otherwise, and
   unlike VRM keys they are not content-hashed and reused, so they would
   accumulate forever. `motions-pinned/` deliberately gets NO rule: a future
   "user keeps this motion" feature is meant to be a `CopyObject` into that
   prefix, not an edit to a lifecycle rule that is already running against
   live data.

2. **A dedicated `motions/*` cache behavior requiring a CloudFront signed
   URL** (trusted key group), because these files are per-job and not meant
   to be publicly guessable/fetchable the way a shared VRM model is. The
   behavior also disables 404 caching (`error_caching_min_ttl=0`): a frontend
   that polls and fetches before the worker's upload lands would otherwise
   get its 404 cached by CloudFront, poisoning that key even after the file
   shows up in S3.

The public key used to verify signed URLs is passed as CDK context
(`motion_public_key_pem`), not read from SSM inside this stack. When it is
absent this stack reports a **stack-scoped** synth error via
`Annotations.of(self).add_error(...)`, not a Python `raise`. `app.py` builds
every stack unconditionally on every `cdk` invocation (`cdk.json`'s `app`
entry is `python app.py`, run for `cdk list`, `cdk diff`, `cdk deploy
VvaVpcStack` — anything), so a `raise` here would crash commands that have
nothing to do with this stack. `Annotations.add_error` is the mechanism CDK
built for exactly this: the CLI fails `cdk synth`/`cdk deploy` for the stack
that has the error, and leaves every other stack's commands working. See
app.py's comment above `CrudApiStack(...)` for the same "unconditional
construction" constraint solved a different way (a public default value)
that doesn't apply here because there is no safe default for a signing key.

An annotation alone doesn't stop Python from continuing to run this
constructor, and `cloudfront.PublicKey` cannot accept a missing key. So when
`motion_public_key_pem` is absent, the key group and the `motions/*` behavior
are skipped entirely (see the `if motion_public_key_pem:` guard below) rather
than built from a fabricated placeholder — this stack has no fake key
material anywhere in it, and still produces a valid (VRM-only) template so
the rest of the app keeps synthesizing.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    Annotations,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_s3 as s3,
)
from constructs import Construct

from infra.origins import resolve as resolve_origins


def _resolve_public_key(stack: Stack, ctx) -> str | None:
    """Return the signing public key PEM, or None if it is missing or unusable.

    Two ways in, because one of them does not work everywhere:

    - ``-c motion_public_key_file=<path>`` — the reliable one. A PEM is
      multi-line, and on Windows a multi-line value does not survive argv: the
      CLI received only ``-----BEGIN PUBLIC KEY-----`` and CDK synthesised a
      26-character key. Synth passed (nothing here looked at the shape) and
      CloudFront rejected it mid-deploy with "empty/invalid/out of limits RSA
      Encoded Key", rolling the stack back.
    - ``-c motion_public_key_pem=<pem>`` — kept because it is what a POSIX
      shell can do in one line, and what the tests inject.

    Line endings are normalised to LF. openssl on Windows writes CRLF, and the
    encoded key must not carry it.

    The shape check is deliberately shallow — a BEGIN line, an END line, and a
    body between them. It is not validating cryptography; it is catching
    truncation, which is the failure that actually happened. A wrong-but-
    well-formed key still fails, but it fails at deploy where CloudFront can
    say so precisely.
    """
    pem = ctx("motion_public_key_pem")
    key_file = ctx("motion_public_key_file")

    if key_file and not pem:
        path = Path(key_file)
        if not path.is_file():
            Annotations.of(stack).add_error(
                f"motion_public_key_file points at {key_file!r}, which is not a "
                "file. Pass the path to the PEM written by:\n"
                "  openssl rsa -in motion_signing_key.pem -pubout "
                "-out motion_signing_key.pub"
            )
            return None
        pem = path.read_text(encoding="utf-8")

    if not pem:
        return None

    pem = pem.replace("\r\n", "\n").replace("\r", "\n").strip()

    if "\n" not in pem:
        Annotations.of(stack).add_error(
            "motion_public_key_pem arrived as a single line, so it is a "
            "truncated PEM, not a key. A multi-line -c value does not survive "
            "argv on Windows — the shell hands over only the first line and "
            "CloudFront rejects it mid-deploy with 'empty/invalid/out of "
            "limits RSA Encoded Key', after the stack has started updating.\n"
            "Pass the path instead:\n"
            "  cdk deploy VvaAssetStack -c motion_public_key_file=path/to/"
            "motion_signing_key.pub"
        )
        return None

    if not (pem.startswith("-----BEGIN PUBLIC KEY-----")
            and pem.endswith("-----END PUBLIC KEY-----")):
        Annotations.of(stack).add_error(
            "motion_public_key_pem is not an SPKI public key PEM: it must "
            "start with '-----BEGIN PUBLIC KEY-----' and end with "
            "'-----END PUBLIC KEY-----'. A private key, an SSH-format key "
            "(ssh-rsa AAAA...) or a certificate will all be rejected by "
            "CloudFront at deploy time.\n"
            "Produce the right one with:\n"
            "  openssl rsa -in motion_signing_key.pem -pubout "
            "-out motion_signing_key.pub"
        )
        return None

    return pem + "\n"


class AssetStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        motion_public_key_pem = _resolve_public_key(self, ctx)

        # No fallback default (unlike the Cognito ids in crud_api_stack.py):
        # those are public, checked-in-anyway values; a signing key is not.
        # Deploying with the wrong one silently, or omitting it and letting
        # CDK invent something, both fail worse than refusing to synth.
        #
        # Annotations.add_error, NOT raise: app.py constructs this stack
        # unconditionally on every `cdk` invocation, so raising here would
        # break `cdk list`, `cdk diff`, `cdk deploy VvaVpcStack` — commands
        # with nothing to do with motions. add_error fails cdk synth/deploy
        # for THIS stack only; the CLI checks each target stack's own
        # annotations. See the module docstring for the full reasoning.
        if not motion_public_key_pem:
            Annotations.of(self).add_error(
                "VvaAssetStack needs the CloudFront signing public key for "
                "motions/*. Pass the PATH, not the bytes — a multi-line "
                "-c motion_public_key_pem value does not survive argv on "
                "Windows and arrives truncated:\n"
                "  cdk deploy VvaAssetStack -c motion_public_key_file="
                "motion_signing_key.pub\n"
                "This verifies signed URLs the agent hands out for GPU-rendered "
                "motion files; without it those files would need to be public, "
                "which they are not meant to be."
            )

        allowed_origins = resolve_origins(self.node)

        # ── Bucket ──────────────────────────────────────────────────────

        self.bucket = s3.Bucket(
            self, "AssetsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,   # keys are content-hashed; versioning adds nothing
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )

        # Ephemeral motion renders expire after 1 day. Scoped to the `motions/`
        # prefix only — everything else in this bucket is a VRM model asset that
        # must never expire. `motions-pinned/` intentionally has no rule; see the
        # module docstring for why.
        self.bucket.add_lifecycle_rule(
            id="ExpireEphemeralMotions",
            prefix="motions/",
            expiration=Duration.days(1),
            enabled=True,
        )

        # ── Response headers (CORS) ─────────────────────────────────────

        cors_policy = cloudfront.ResponseHeadersPolicy(
            self, "AssetCorsPolicy",
            comment="CORS for VRM assets and the character catalog",
            cors_behavior=cloudfront.ResponseHeadersCorsBehavior(
                access_control_allow_credentials=False,
                access_control_allow_headers=["*"],
                access_control_allow_methods=["GET", "HEAD", "OPTIONS"],
                access_control_allow_origins=allowed_origins,
                access_control_expose_headers=["ETag", "Content-Length", "Content-Type"],
                access_control_max_age=Duration.hours(1),
                origin_override=True,
            ),
        )

        # ── Signed-URL key group (motions/* only) ───────────────────────
        #
        # Registers the public key CloudFront verifies signed URLs against.
        # The agent signs URLs with the matching private key (kept outside CDK
        # entirely — it never appears here) when it hands a rendered motion
        # back to a caller.
        #
        # Guarded on the key being present: with it absent, the Annotations
        # error above already blocks `cdk deploy`/`cdk synth` for this stack,
        # so there is nothing to gain from inventing a placeholder key here —
        # and every alternative to skipping is either a jsii TypeError (None)
        # or fabricated key material that shouldn't exist in this file at all.
        self.motion_key_group = None
        if motion_public_key_pem:
            motion_public_key = cloudfront.PublicKey(
                self, "MotionSigningKey",
                encoded_key=motion_public_key_pem,
            )
            self.motion_key_group = cloudfront.KeyGroup(
                self, "MotionKeyGroup",
                items=[motion_public_key],
            )

        # ── Distribution ────────────────────────────────────────────────

        # motions/* is its own behavior, not folded into the default one,
        # because it is the only path that needs a trusted key group — a
        # signed-URL requirement on the VRM paths would break every existing
        # caller of the unsigned model URLs. Only added when the key group
        # exists; see the guard above.
        additional_behaviors = {}
        if self.motion_key_group is not None:
            additional_behaviors["motions/*"] = cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                response_headers_policy=cors_policy,
                compress=True,   # .bvh is plain text and compresses well
                trusted_key_groups=[self.motion_key_group],
            )

        self.distribution = cloudfront.Distribution(
            self, "AssetDistribution",
            comment="VVA assets (VRM models) + character catalog",
            # PRICE_CLASS_200 covers North America, Europe and Asia — including
            # the users this is built for. The default (all edge locations) adds
            # South America, Australia and Africa at a higher per-GB rate for
            # traffic this project does not have.
            price_class=cloudfront.PriceClass.PRICE_CLASS_200,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                response_headers_policy=cors_policy,
                compress=True,
            ),
            additional_behaviors=additional_behaviors,
            error_responses=[
                # Motivated by motions/*: a frontend that polls and fetches a
                # motion URL before the GPU worker's upload lands gets a 404
                # that CACHING_OPTIMIZED would otherwise cache for its default
                # TTL, poisoning that key at the edge even after the file
                # shows up in S3 moments later.
                #
                # CloudFront has no per-behavior error-response setting —
                # error_responses is DISTRIBUTION-WIDE, so this also disables
                # 404 caching on the default (VRM) behavior. That's accepted:
                # VRM keys are content-hashed and requested only after the
                # upload script has already put the object in S3, so a VRM
                # 404 recovering instantly rather than staying cached is
                # harmless — there's no legitimate case where a VRM 404 is
                # expected to resolve into a 200 without a new key/deploy.
                cloudfront.ErrorResponse(http_status=404, ttl=Duration.seconds(0)),
            ],
        )

        # A CfnPermission used to sit here, granting CloudFront
        # `lambda:InvokeFunction` on the characters function — the grant that
        # FunctionUrlOrigin.with_origin_access_control() does not add and that
        # Lambda has required since October 2025. It went away with the Function
        # URL origin it existed for: the catalog is served by VvaRestApiStack now,
        # and this distribution has no Lambda origin left.

        # ── Outputs ─────────────────────────────────────────────────────

        CfnOutput(
            self, "AssetBucketName",
            value=self.bucket.bucket_name,
            description="Upload target for scripts/upload_characters_to_s3.py",
        )
        CfnOutput(
            self, "AssetBaseUrl",
            value=f"https://{self.distribution.distribution_domain_name}",
            description="Frontend VITE_ASSET_BASE_URL",
        )
