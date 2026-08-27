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
(`motion_public_key_pem`), not read from SSM inside this stack — see the
`raise ValueError` below and crud_api_stack.py's Cognito-id check for the
pattern this follows: a required deploy-time input that is missing must fail
at synth, not deploy something silently wrong.
"""

from __future__ import annotations

from aws_cdk import (
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


class AssetStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        motion_public_key_pem = ctx("motion_public_key_pem")

        # No fallback default (unlike the Cognito ids in crud_api_stack.py):
        # those are public, checked-in-anyway values; a signing key is not.
        # Deploying with the wrong one silently, or omitting it and letting
        # CDK invent something, both fail worse than refusing to synth.
        if not motion_public_key_pem:
            raise ValueError(
                "VvaAssetStack needs the CloudFront signing public key for "
                "motions/*. Pass:\n"
                "  cdk deploy VvaAssetStack -c motion_public_key_pem=\"$(cat "
                "motion_signing_key.pub)\"\n"
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
        motion_public_key = cloudfront.PublicKey(
            self, "MotionSigningKey",
            encoded_key=motion_public_key_pem,
        )
        self.motion_key_group = cloudfront.KeyGroup(
            self, "MotionKeyGroup",
            items=[motion_public_key],
        )

        # ── Distribution ────────────────────────────────────────────────

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
            # motions/* is its own behavior, not folded into the default one,
            # because it is the only path that needs a trusted key group — a
            # signed-URL requirement on the VRM paths would break every
            # existing caller of the unsigned model URLs.
            additional_behaviors={
                "motions/*": cloudfront.BehaviorOptions(
                    origin=origins.S3BucketOrigin.with_origin_access_control(self.bucket),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                    cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                    response_headers_policy=cors_policy,
                    compress=True,   # .bvh is plain text and compresses well
                    trusted_key_groups=[self.motion_key_group],
                ),
            },
            error_responses=[
                # Without this, a frontend that polls and fetches a motion URL
                # before the GPU worker's upload lands gets a 404 that
                # CACHING_OPTIMIZED would otherwise cache for its default TTL —
                # poisoning that key at the edge even after the file shows up
                # in S3 moments later.
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
