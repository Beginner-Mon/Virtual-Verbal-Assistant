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

# The deployed frontend must be here, not only the dev origins. A CORS list that
# covers localhost and nothing else fails exactly once — in production, after the
# release, with the browser reporting a blocked fetch rather than anything that
# points at this file. Override per deploy with
# `-c allowed_origins=https://a,https://b`, but the default has to be correct on
# its own: a context flag is lost the moment someone deploys without it.
_DEFAULT_ALLOWED_ORIGINS = [
    "https://release.d32nf9wwqqt016.amplifyapp.com",
    "http://localhost:5173",
    "http://localhost:3000",
]


class AssetStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Comma-separated, e.g. -c allowed_origins=https://app.example.com,http://localhost:5173
        raw_origins = self.node.try_get_context("allowed_origins")
        allowed_origins = (
            [o.strip() for o in raw_origins.split(",") if o.strip()]
            if raw_origins else list(_DEFAULT_ALLOWED_ORIGINS)
        )

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
