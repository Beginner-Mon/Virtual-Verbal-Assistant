"""VPC Stack — project-vva network foundation.

Creates a VPC with 4 subnets across 2 AZs:
  - 2 'EcsPublic' subnets (PUBLIC + IGW, auto-assign public IP, for ECS GPU tasks)
  - 2 'Private' subnets (PRIVATE_ISOLATED, for Lambda + RDS)

VPC Endpoints for SSM Parameter Store, STS, and CloudWatch Logs enable Lambda
functions in isolated subnets to access these AWS services without internet
connectivity. S3 Gateway Endpoint on EcsPublic subnets for free internal S3 access.

To add NAT Gateway later:
  - Change 'Private' subnet_type to ec2.SubnetType.PRIVATE_WITH_EGRESS
  - Set nat_gateways=1 (or 2 for HA)
"""

from aws_cdk import (
    Stack,
    Tags,
    CfnOutput,
    aws_ec2 as ec2,
)
from constructs import Construct


class VpcStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── VPC ─────────────────────────────────────────────────────────
        self.vpc = ec2.Vpc(
            self, "ProjectVva",
            vpc_name="project-vva",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            max_azs=2,
            nat_gateways=0,  # No NAT — add later if Lambda needs internet
            subnet_configuration=[
                # PUBLIC subnets for ECS GPU tasks (Kimodo MCP).
                # IGW auto-created by CDK when subnet_type=PUBLIC.
                ec2.SubnetConfiguration(
                    name="EcsPublic",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                    map_public_ip_on_launch=True,
                ),
                # Lambda functions, RDS, RDS Proxy live here.
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        # ── VPC Interface Endpoints ─────────────────────────────────────
        # Lambda in isolated subnets has NO internet path. These endpoints
        # create private links to AWS services within the VPC so Lambda can:
        #   - Read DB connection config from SSM Parameter Store
        #   - Generate IAM auth tokens via STS (used by boto3 RDS client)
        #   - Push execution logs to CloudWatch

        # self.sg_vpc_endpoints = ec2.SecurityGroup(
        #     self, "SgVpcEndpoints",
        #     vpc=self.vpc,
        #     security_group_name="vva-sg-vpc-endpoints",
        #     description="Allow HTTPS from within VPC to Interface Endpoints",
        #     allow_all_outbound=False,
        # )
        # self.sg_vpc_endpoints.add_ingress_rule(
        #     peer=ec2.Peer.ipv4(self.vpc.vpc_cidr_block),
        #     connection=ec2.Port.tcp(443),
        #     description="HTTPS from VPC CIDR",
        # )

        # SSM Parameter Store — Lambda reads DB connection params
        # self.vpc.add_interface_endpoint(
        #     "SsmEndpoint",
        #     service=ec2.InterfaceVpcEndpointAwsService.SSM,
        #     subnets=ec2.SubnetSelection(subnet_group_name="Private"),
        #     security_groups=[self.sg_vpc_endpoints],
        # )

        # STS — Lambda generates IAM auth token for RDS Proxy
        # self.vpc.add_interface_endpoint(
        #     "StsEndpoint",
        #     service=ec2.InterfaceVpcEndpointAwsService.STS,
        #     subnets=ec2.SubnetSelection(subnet_group_name="Private"),
        #     security_groups=[self.sg_vpc_endpoints],
        # )

        # S3 Gateway Endpoint — free, route-table level. EC2 User Data uses
        # this to download SMPLX_NEUTRAL.npz without internet.
        self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[ec2.SubnetSelection(subnet_group_name="EcsPublic")],
        )

        # ── Exports kept alive across the Kimodo queue migration ────────
        #
        # VvaKimodoEcsStack used to run an ALB, a target group and an ECS
        # Service, all of which needed subnets and the VPC CIDR. The queue
        # design has none of them, so CDK stopped emitting these exports —
        # correct for the new template, and un-deployable from where the
        # account is: CloudFormation will not delete an export while another
        # stack imports it, and the DEPLOYED Kimodo stack still does. `cdk
        # deploy VvaVpcStack VvaKimodoEcsStack` does the VPC first, so the
        # removal fails and the VPC rolls back before Kimodo is ever reached.
        #
        # export_value pins them so this deploy can land. They come out in a
        # LATER deploy, once the new Kimodo stack is live and nothing imports
        # them — delete these lines then, together with
        # tests/infra/test_vpc_export_retention.py.
        self.export_value(self.vpc.vpc_cidr_block)
        for subnet in (*self.vpc.public_subnets, *self.vpc.isolated_subnets):
            self.export_value(subnet.subnet_id)

        # ── Tags ─────────────────────────────────────────────────────────
        Tags.of(self).add("Project", "ECA")

        # ── Outputs ─────────────────────────────────────────────────────
        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
