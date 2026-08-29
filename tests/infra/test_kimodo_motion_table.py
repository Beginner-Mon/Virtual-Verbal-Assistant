import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from infra.kimodo_ecs_stack import KimodoEcsStack
from infra.vpc_stack import VpcStack


@pytest.fixture
def template():
    app = cdk.App()
    env = cdk.Environment(account="244203483654", region="us-east-1")
    vpc_stack = VpcStack(app, "Vpc", env=env)
    stack = KimodoEcsStack(
        app, "Kimodo",
        vpc=vpc_stack.vpc,
        assets_bucket_name="vva-test-assets-bucket",
        env=env,
    )
    return Template.from_stack(stack)


@pytest.mark.unit
def test_table_has_gsi_and_ttl(template):
    template.has_resource_properties("AWS::DynamoDB::Table", Match.object_like({
        "BillingMode": "PAY_PER_REQUEST",
        "TimeToLiveSpecification": {"AttributeName": "expires_at", "Enabled": True},
        "GlobalSecondaryIndexes": Match.array_with([
            Match.object_like({"IndexName": "status-created_at-index"})
        ]),
    }))


@pytest.mark.unit
def test_dynamodb_gateway_endpoint_exists(template):
    """Gateway endpoint (free), NOT an interface endpoint (billed hourly)."""
    template.has_resource_properties("AWS::EC2::VPCEndpoint", Match.object_like({
        "VpcEndpointType": "Gateway",
    }))


@pytest.mark.unit
def test_no_load_balancer(template):
    """ALB is gone entirely — that is the whole point of this architecture."""
    template.resource_count_is("AWS::ElasticLoadBalancingV2::LoadBalancer", 0)
