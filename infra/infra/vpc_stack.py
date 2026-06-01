"""VPC Stack — project-vva network foundation.

Creates a VPC with 4 isolated subnets (2 'Public' reserved for future IGW/NAT,
2 'Private' for Lambda/RDS). No Internet Gateway, no NAT Gateway.

VPC Endpoints for SSM Parameter Store, STS, and CloudWatch Logs enable Lambda
functions in isolated subnets to access these AWS services without internet
connectivity. SSM provides DB connection config, STS is used internally by
boto3 to generate IAM auth tokens for RDS Proxy.

To add IGW later:
  - Change 'Public' subnet_type to ec2.SubnetType.PUBLIC
  - CDK will auto-create IGW + route table entries

To add NAT Gateway later:
  - Change 'Private' subnet_type to ec2.SubnetType.PRIVATE_WITH_EGRESS
  - Set nat_gateways=1 (or 2 for HA)
"""

from aws_cdk import (
    Stack,
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
                # Reserved for future IGW/ALB/public-facing resources.
                # Currently PRIVATE_ISOLATED because no IGW is provisioned.
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
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

        self.sg_vpc_endpoints = ec2.SecurityGroup(
            self, "SgVpcEndpoints",
            vpc=self.vpc,
            security_group_name="vva-sg-vpc-endpoints",
            description="Allow HTTPS from within VPC to Interface Endpoints",
            allow_all_outbound=False,
        )
        self.sg_vpc_endpoints.add_ingress_rule(
            peer=ec2.Peer.ipv4(self.vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(443),
            description="HTTPS from VPC CIDR",
        )

        # SSM Parameter Store — Lambda reads DB connection params
        self.vpc.add_interface_endpoint(
            "SsmEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SSM,
            subnets=ec2.SubnetSelection(subnet_group_name="Private"),
            security_groups=[self.sg_vpc_endpoints],
        )

        # STS — Lambda generates IAM auth token for RDS Proxy
        self.vpc.add_interface_endpoint(
            "StsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.STS,
            subnets=ec2.SubnetSelection(subnet_group_name="Private"),
            security_groups=[self.sg_vpc_endpoints],
        )

        # CloudWatch Logs — Lambda writes execution logs
        self.vpc.add_interface_endpoint(
            "CloudWatchLogsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            subnets=ec2.SubnetSelection(subnet_group_name="Private"),
            security_groups=[self.sg_vpc_endpoints],
        )

        # ── Outputs ─────────────────────────────────────────────────────
        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
