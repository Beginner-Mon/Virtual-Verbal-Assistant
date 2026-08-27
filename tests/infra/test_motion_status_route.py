"""GET /motion/{job_id} (Task 9) — authed the same way every other data route is.

Motion files are the first user-derived artifacts on the CDN (see
asset_stack.py's module docstring): everything served before this was a
static app asset, uploaded by a script rather than produced by a stranger's
job. That is why this route needs the SAME Cognito authorizer every other
data route in rest_api_stack.py already carries, and why AgentStack needs a
scoped IAM grant on the motion job table it does not construct (ruling R3).
"""

import aws_cdk as cdk
import pytest
from aws_cdk import aws_lambda as lambda_
from aws_cdk.assertions import Match, Template

from infra.agent_stack import AgentStack
from infra.asset_stack import AssetStack
from infra.rest_api_stack import RestApiStack

_ENV = cdk.Environment(account="244203483654", region="us-east-1")

_DUMMY_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAMDummyKeyForTestingPurposesOnly"
    "NeverUseInProductionAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "ECAwEAAQ==\n"
    "-----END PUBLIC KEY-----\n"
)


def _dummy_fn(scope, cid):
    """A minimal Lambda — stands in for crud_fn/characters_fn, neither of
    which this test exercises. RestApiStack only needs an IFunction to wire
    a LambdaIntegration; what it does is irrelevant here."""
    return lambda_.Function(
        scope, cid,
        runtime=lambda_.Runtime.PYTHON_3_12,
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context):\n    return {}"),
    )


@pytest.fixture
def agent_template():
    app = cdk.App(context={
        "motion_public_key_pem": _DUMMY_PEM,
        "agent_image_tag": "deadbeef",
        # R24: no raw secret context flags — motion_hash_secret_param and
        # motion_signing_key_param both fall back to their module defaults,
        # exactly as a real deploy that only overrides motion_key_pair_id
        # would. That is the point of this fixture: prove the stack needs no
        # secret value at synth time at all.
        "motion_key_pair_id": "K2EXAMPLE",
    })
    asset_stack = AssetStack(app, "Assets", env=_ENV)
    agent_stack = AgentStack(
        app, "Agent",
        asset_base_url=f"https://{asset_stack.distribution.distribution_domain_name}",
        env=_ENV,
    )
    return Template.from_stack(agent_stack)


@pytest.fixture
def rest_template():
    app = cdk.App()
    crud_stack = cdk.Stack(app, "DummyDeps", env=_ENV)
    crud_fn = _dummy_fn(crud_stack, "Crud")
    characters_fn = _dummy_fn(crud_stack, "Characters")
    agent_fn = _dummy_fn(crud_stack, "Agent")
    rest_stack = RestApiStack(
        app, "Rest",
        crud_fn=crud_fn,
        characters_fn=characters_fn,
        cognito_pool_id="us-east-1_TESTPOOL",
        agent_fn=agent_fn,
        env=_ENV,
    )
    return Template.from_stack(rest_stack)


@pytest.mark.unit
def test_motion_route_requires_cognito_auth(rest_template):
    rest_template.has_resource_properties("AWS::ApiGateway::Method", Match.object_like({
        "HttpMethod": "GET",
        "AuthorizationType": "COGNITO_USER_POOLS",
        "ResourceId": Match.any_value(),
    }))
    # Narrow to the specific /motion/{job_id} method: find the resource chain
    # motion -> {job_id} and confirm ITS method is the authed GET, not some
    # other route's.
    body = rest_template.to_json()
    resources = body["Resources"]
    motion_res = next(
        rid for rid, r in resources.items()
        if r["Type"] == "AWS::ApiGateway::Resource"
        and r["Properties"].get("PathPart") == "motion"
    )
    job_id_res = next(
        rid for rid, r in resources.items()
        if r["Type"] == "AWS::ApiGateway::Resource"
        and r["Properties"].get("PathPart") == "{job_id}"
        and r["Properties"].get("ParentId", {}).get("Ref") == motion_res
    )
    method = next(
        r for r in resources.values()
        if r["Type"] == "AWS::ApiGateway::Method"
        and r["Properties"].get("ResourceId", {}).get("Ref") == job_id_res
        and r["Properties"].get("HttpMethod") == "GET"
    )
    assert method["Properties"]["HttpMethod"] == "GET"
    assert method["Properties"]["AuthorizationType"] == "COGNITO_USER_POOLS"


@pytest.mark.unit
def test_motion_route_has_no_second_authorizer(rest_template):
    """Task 9's brief is explicit: reuse the authorizer built for /sessions,
    /me/memory, /chat, /tts, /billing — do not construct a second one."""
    rest_template.resource_count_is("AWS::ApiGateway::Authorizer", 1)


@pytest.mark.unit
def test_agent_lambda_gets_motion_table_permissions_via_fixed_arn(agent_template):
    """Ruling R3: KimodoEcsStack (which owns the real Table construct) is
    built after AgentStack in app.py, so the grant here has to be built from
    an ARN using the fixed table name, not `.grant_read_write_data()`."""
    agent_template.has_resource_properties("AWS::IAM::Policy", Match.object_like({
        "PolicyDocument": Match.object_like({
            "Statement": Match.array_with([
                Match.object_like({
                    "Action": Match.array_with(["dynamodb:GetItem"]),
                    "Resource": Match.array_with([
                        Match.string_like_regexp(r".*table/vva-motion-jobs$"),
                        Match.string_like_regexp(r".*table/vva-motion-jobs/index/\*$"),
                    ]),
                }),
            ]),
        }),
    }))


@pytest.mark.unit
def test_agent_lambda_env_carries_motion_config(agent_template):
    agent_template.has_resource_properties("AWS::Lambda::Function", Match.object_like({
        "Environment": {"Variables": Match.object_like({
            "MOTION_TABLE": "vva-motion-jobs",
            "MOTION_HASH_SECRET_PARAM": "/vva/motion/hash-secret",
            "MOTION_SIGNING_KEY_PARAM": "/vva/motion/signing-key-pem",
            "MOTION_KEY_PAIR_ID": "K2EXAMPLE",
            # A CDK token (Fn::Join over the distribution's domain name), not
            # a literal string yet — it only resolves at deploy. Presence is
            # what matters here; test_agent_lambda_gets_motion_table_... and
            # the app.py wiring itself prove it's wired to the real value.
            "ASSET_BASE_URL": Match.any_value(),
        })},
    }))


@pytest.mark.unit
def test_signing_key_param_never_holds_the_key_itself(agent_template):
    """The private key must never be embedded as a literal value anywhere in
    this template — only its SSM parameter NAME. asset_stack.py's docstring
    calls the matching public key "kept outside CDK entirely"; the private
    half is the same class of secret, one degree more sensitive."""
    body = agent_template.to_json()
    serialized = str(body)
    assert "BEGIN RSA PRIVATE KEY" not in serialized
    assert "BEGIN PRIVATE KEY" not in serialized


@pytest.mark.unit
def test_hash_secret_never_lands_in_template_plaintext(agent_template):
    """Ruling R24: MOTION_HASH_SECRET (a raw value) must never be a Lambda
    env var — only MOTION_HASH_SECRET_PARAM (the SSM parameter NAME) is.
    Mirrors test_signing_key_param_never_holds_the_key_itself's negative
    assertion, for the second of the brief's three "sensitive, from SSM"
    values. An earlier version of this stack put the raw secret behind a
    `motion_hash_secret` CDK context flag, which this test would have caught:
    that flag is gone entirely now, not just unused by this fixture."""
    body = agent_template.to_json()
    fn = next(
        r for r in body["Resources"].values()
        if r["Type"] == "AWS::Lambda::Function"
    )
    env_vars = fn["Properties"]["Environment"]["Variables"]
    assert "MOTION_HASH_SECRET" not in env_vars
    assert env_vars["MOTION_HASH_SECRET_PARAM"] == "/vva/motion/hash-secret"
