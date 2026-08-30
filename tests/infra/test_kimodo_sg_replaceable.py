"""The Kimodo task security group must not carry a fixed name.

A security group's Description is immutable in CloudFormation: changing it
replaces the group. This branch changed it from "Kimodo MCP - inbound 8000" to
one describing the queue worker, which is accurate and cost a rolled-back
deploy:

    Resource handler returned message: "Security Group with kimodo-sg-ecs
    already exists" (HandlerErrorCode: AlreadyExists)

Replacement creates the new group before deleting the old one, so a fixed
`security_group_name` guarantees the pair collides. The two settings together —
an immutable property that is edited whenever the design changes, and a name
that makes replacing it impossible — mean every future edit to that description
is an un-deployable change. Dropping the explicit name is what makes the group
replaceable at all; CDK's generated name is unique per deployment.

Nothing outside the stack referred to `kimodo-sg-ecs`, and nothing should: with
no ECS Service, the operator passes the group to `aws ecs run-task`, and the id
belongs in a stack output rather than a name people retype.
"""

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

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
def test_task_security_group_has_no_fixed_name(template):
    groups = [
        r["Properties"]
        for r in template.to_json()["Resources"].values()
        if r["Type"] == "AWS::EC2::SecurityGroup"
    ]
    assert groups, "expected the task security group"
    for props in groups:
        assert "GroupName" not in props, (
            "A fixed GroupName makes the group unreplaceable, and its "
            "Description is immutable — so editing the description becomes an "
            "un-deployable change. Let CDK generate the name."
        )
