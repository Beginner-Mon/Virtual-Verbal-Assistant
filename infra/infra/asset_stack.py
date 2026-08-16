"""Asset Stack — Track 2: private S3 + CloudFront for VRM models and the catalog.

    https://<dist>/characters*              → Lambda Function URL (CharacterStack)
    https://<dist>/characters/{slug}/*.vrm  → S3

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
    aws_lambda as lambda_,
    aws_s3 as s3,
)
from constructs import Construct

_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]


class AssetStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        characters_fn_url: lambda_.IFunctionUrl,
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

        # ── Cache policy for the catalog ────────────────────────────────
        # Short TTL rather than CACHING_OPTIMIZED: the catalog has no hash in
        # its URL, so a seeded change has to become visible without an
        # invalidation. Five minutes still collapses essentially all traffic.
        #
        # If per-user gating is ever added to /characters, this must become
        # cache-by-header or no-cache — a shared edge cache cannot serve
        # responses that differ per viewer.

        catalog_cache = cloudfront.CachePolicy(
            self, "CatalogCachePolicy",
            comment="Character catalog — short TTL, no cookies or auth headers",
            default_ttl=Duration.minutes(5),
            min_ttl=Duration.seconds(0),
            max_ttl=Duration.minutes(15),
            enable_accept_encoding_gzip=True,
            enable_accept_encoding_brotli=True,
            header_behavior=cloudfront.CacheHeaderBehavior.none(),
            query_string_behavior=cloudfront.CacheQueryStringBehavior.none(),
            cookie_behavior=cloudfront.CacheCookieBehavior.none(),
        )

        # ── Distribution ────────────────────────────────────────────────

        self.distribution = cloudfront.Distribution(
            self, "AssetDistribution",
            comment="VVA assets (VRM models) + character catalog",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                response_headers_policy=cors_policy,
                compress=True,
            ),
            additional_behaviors={
                # Matches /characters, /characters/bronya and
                # /characters/bronya/avatar-profile alike.
                "/characters*": cloudfront.BehaviorOptions(
                    origin=origins.FunctionUrlOrigin.with_origin_access_control(
                        characters_fn_url
                    ),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                    cache_policy=catalog_cache,
                    response_headers_policy=cors_policy,
                    compress=True,
                ),
            },
        )

        # ── Lambda permission that CDK does not add ─────────────────────
        #
        # FunctionUrlOrigin.with_origin_access_control() grants CloudFront
        # `lambda:InvokeFunctionUrl` and stops there. Since October 2025 a
        # function URL also requires `lambda:InvokeFunction`, so OAC requests
        # arrive signed and correct and Lambda rejects them anyway — 403 with
        # the generic "Function URL authorization" message and nothing in the
        # function's logs, because the rejection happens before invocation.
        #
        # Everything else looks right while this is missing, which is what makes
        # it expensive to find: the OAC is type `lambda`, signing is `always`,
        # the resource policy names the distribution, and a request signed by
        # hand with an admin identity succeeds — an admin has both actions.
        #
        # Remove once aws-cdk-lib grants both. Tracked upstream via
        # aws-samples/remote-swe-agents#361.
        lambda_.CfnPermission(
            self, "CharactersInvokeFunctionFromCloudFront",
            action="lambda:InvokeFunction",
            function_name=characters_fn_url.function_arn,
            principal="cloudfront.amazonaws.com",
            source_arn=(
                f"arn:aws:cloudfront::{self.account}:distribution/"
                f"{self.distribution.distribution_id}"
            ),
        )

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
