"""VvaVpcStack must keep publishing the exports the deployed VvaKimodoEcsStack
still imports, even though the new Kimodo template no longer uses them.

Dropping the ALB, the target group and the ECS Service from VvaKimodoEcsStack
removed the only consumers of five VvaVpcStack exports, so CDK stopped emitting
them. That synthesises cleanly and cannot deploy: CloudFormation refuses to
delete an export while another stack still imports it, and the *deployed*
Kimodo stack does until its own update lands. `cdk deploy VvaVpcStack
VvaKimodoEcsStack` deploys the VPC first — the export removal fails, and the
VPC stack rolls back before Kimodo is ever reached.

Nothing caught this: the CDK tests synthesise templates and never see deployed
state, and the synthesised template is not wrong — it is only un-deployable
from where the account actually is.

`export_value` is CDK's documented answer. It pins the export so the value
survives a deploy in which nothing references it. It comes out in a *later*
deploy, once the new Kimodo stack is live and the imports are gone — a
follow-up, not part of this migration.
"""

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from infra.vpc_stack import VpcStack

# Read off `aws cloudformation list-exports` for the live account. The names are
# CDK-generated from logical ids, so they are stable, and they are what the
# deployed Kimodo stack's Fn::ImportValue calls actually name.
_IN_USE_EXPORTS = [
    "VvaVpcStack:ExportsOutputFnGetAttProjectVva2EF6510DCidrBlockF687B416",
    "VvaVpcStack:ExportsOutputRefProjectVvaEcsPublicSubnet1Subnet802161998B50E054",
    "VvaVpcStack:ExportsOutputRefProjectVvaEcsPublicSubnet2Subnet5ACCA079A7EF8308",
    "VvaVpcStack:ExportsOutputRefProjectVvaPrivateSubnet1Subnet7B8261A126E7A255",
    "VvaVpcStack:ExportsOutputRefProjectVvaPrivateSubnet2Subnet89E2EA3BF5D4685A",
]


@pytest.fixture
def exported_names():
    app = cdk.App()
    stack = VpcStack(
        app, "VvaVpcStack",
        env=cdk.Environment(account="244203483654", region="us-east-1"),
    )
    outputs = Template.from_stack(stack).to_json().get("Outputs", {})
    return {
        o["Export"]["Name"]
        for o in outputs.values()
        if isinstance(o.get("Export", {}).get("Name"), str)
    }


@pytest.mark.unit
@pytest.mark.parametrize("name", _IN_USE_EXPORTS)
def test_export_still_published(name, exported_names):
    assert name in exported_names, (
        f"{name} is imported by the deployed VvaKimodoEcsStack. Removing it "
        "makes VvaVpcStack un-deployable until that import is gone; keep the "
        "matching export_value() call until a follow-up deploy drops it."
    )
