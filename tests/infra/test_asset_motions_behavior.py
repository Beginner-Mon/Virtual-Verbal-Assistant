"""Motion files (Task 7) are the first user-derived artifacts on the CDN AssetStack
serves — everything else on it is static app assets. That is why they get an
expiry lifecycle rule and a signed-URL cache behavior and the VRM files do not.
See infra/infra/asset_stack.py's module docstring for the full picture.
"""

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from infra.asset_stack import AssetStack

_DUMMY_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAMDummyKeyForTestingPurposesOnly"
    "NeverUseInProductionAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "ECAwEAAQ==\n"
    "-----END PUBLIC KEY-----\n"
)


@pytest.fixture
def template():
    app = cdk.App(context={"motion_public_key_pem": _DUMMY_PEM})
    stack = AssetStack(
        app, "Assets",
        env=cdk.Environment(account="244203483654", region="us-east-1"),
    )
    return Template.from_stack(stack)


@pytest.mark.unit
def test_motions_prefix_expires_in_one_day(template):
    template.has_resource_properties("AWS::S3::Bucket", Match.object_like({
        "LifecycleConfiguration": {"Rules": Match.array_with([
            Match.object_like({"Prefix": "motions/", "ExpirationInDays": 1, "Status": "Enabled"})
        ])}
    }))


@pytest.mark.unit
def test_pinned_prefix_has_no_rule(template):
    """motions-pinned/ deliberately has NO rule — so a future "keep this motion"
    feature is a CopyObject into that prefix, not an edit to a running rule."""
    body = template.to_json()
    resources = body["Resources"]
    rule_groups = [
        r["Properties"]["LifecycleConfiguration"]["Rules"]
        for r in resources.values()
        if r["Type"] == "AWS::S3::Bucket" and "LifecycleConfiguration" in r.get("Properties", {})
    ]
    prefixes = [rule.get("Prefix") for group in rule_groups for rule in group]
    assert "motions-pinned/" not in prefixes


@pytest.mark.unit
def test_motions_behavior_requires_signed_url_and_no_404_cache(template):
    template.has_resource_properties("AWS::CloudFront::Distribution", Match.object_like({
        "DistributionConfig": Match.object_like({
            "CacheBehaviors": Match.array_with([
                Match.object_like({
                    "PathPattern": "motions/*",
                    "TrustedKeyGroups": Match.any_value(),
                })
            ]),
            "CustomErrorResponses": Match.array_with([
                Match.object_like({"ErrorCode": 404, "ErrorCachingMinTTL": 0})
            ]),
        })
    }))


@pytest.mark.unit
def test_missing_motion_public_key_raises_at_synth():
    """R17: the signing public key comes from CDK context (following
    crud_api_stack.py's Cognito-id pattern), not an __init__ parameter read from
    SSM. A missing required input must fail at synth rather than deploy something
    silently wrong (see agent_stack.py's docstring for the same rule)."""
    app = cdk.App()
    with pytest.raises(ValueError, match="motion_public_key_pem"):
        AssetStack(
            app, "Assets",
            env=cdk.Environment(account="244203483654", region="us-east-1"),
        )
