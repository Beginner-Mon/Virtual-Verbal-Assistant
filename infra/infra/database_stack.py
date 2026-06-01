"""Database Stack - RDS PostgreSQL + Secrets Manager + RDS Proxy + SSM.

Single-AZ RDS PostgreSQL 16 instance (vva-db) with:
- IAM Authentication enabled (Lambda uses short-lived tokens)
- Credentials managed by AWS Secrets Manager (for admin/rotation)
- RDS Proxy with TLS enforcement, IAM auth, and 30% max connections
- SSM Parameter Store for DB connection config (host, port, db, user)
- Security groups: sg-rds-proxy → sg-rds (chain)

max_connections_percent=30:
  db.t4g.micro has ~87 max_connections (1GB RAM).
  30% ≈ 26 concurrent connections - sufficient for Lambda at MVP scale.
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    CfnOutput,
    aws_ec2 as ec2,
    aws_rds as rds,
    aws_ssm as ssm,
)
from constructs import Construct


class DatabaseStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.Vpc,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Security Groups ─────────────────────────────────────────────

        # RDS instance - only accepts traffic from RDS Proxy
        self.sg_rds = ec2.SecurityGroup(
            self, "SgRds",
            vpc=vpc,
            security_group_name="vva-sg-rds",
            description="RDS PostgreSQL - inbound from RDS Proxy only",
            allow_all_outbound=False,
        )

        # RDS Proxy - accepts traffic from Lambda, forwards to RDS
        self.sg_rds_proxy = ec2.SecurityGroup(
            self, "SgRdsProxy",
            vpc=vpc,
            security_group_name="vva-sg-rds-proxy",
            description="RDS Proxy - inbound from Lambda, outbound to RDS",
            allow_all_outbound=False,
        )

        # Chain: Proxy → RDS (outbound on proxy, inbound on RDS)
        self.sg_rds.add_ingress_rule(
            peer=self.sg_rds_proxy,
            connection=ec2.Port.tcp(5432),
            description="PostgreSQL from RDS Proxy",
        )
        self.sg_rds_proxy.add_egress_rule(
            peer=self.sg_rds,
            connection=ec2.Port.tcp(5432),
            description="PostgreSQL to RDS instance",
        )

        # ── RDS Instance ────────────────────────────────────────────────

        self.db_instance = rds.DatabaseInstance(
            self, "VvaDb",
            instance_identifier="vva-db",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_16,
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO,
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_group_name="Private"),
            security_groups=[self.sg_rds],
            multi_az=False,                         # Single-AZ for dev/MVP
            allocated_storage=20,
            storage_type=rds.StorageType.GP3,
            database_name="vva",
            iam_authentication=True,                 # Enable IAM auth
            # Secrets Manager auto-generates and stores credentials.
            # Secret is still needed for initial setup, rotation, and admin.
            credentials=rds.Credentials.from_generated_secret(
                "vva_admin",
                secret_name="vva-db-credentials",
            ),
            removal_policy=RemovalPolicy.DESTROY,   # Dev - RETAIN for prod
            deletion_protection=False,               # Dev - enable for prod
            backup_retention=Duration.days(7),
        )

        # ── RDS Proxy ───────────────────────────────────────────────────

        self.rds_proxy = rds.DatabaseProxy(
            self, "VvaDbProxy",
            db_proxy_name="vva-db-proxy",
            proxy_target=rds.ProxyTarget.from_instance(self.db_instance),
            secrets=[self.db_instance.secret],
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_group_name="Private"),
            security_groups=[self.sg_rds_proxy],
            require_tls=True,
            iam_auth=True,
            max_connections_percent=30,              # ~26 connections on db.t4g.micro
            idle_client_timeout=Duration.minutes(5),
        )

        # ── SSM Parameter Store (DB connection config for Lambda) ──────

        self.db_param_prefix = "/vva/db"

        ssm.StringParameter(
            self, "ParamDbHost",
            parameter_name=f"{self.db_param_prefix}/proxy_endpoint",
            string_value=self.rds_proxy.endpoint,
            description="RDS Proxy endpoint for VVA",
        )
        ssm.StringParameter(
            self, "ParamDbPort",
            parameter_name=f"{self.db_param_prefix}/port",
            string_value="5432",
            description="RDS PostgreSQL port",
        )
        ssm.StringParameter(
            self, "ParamDbName",
            parameter_name=f"{self.db_param_prefix}/name",
            string_value="vva",
            description="Database name",
        )
        ssm.StringParameter(
            self, "ParamDbUser",
            parameter_name=f"{self.db_param_prefix}/username",
            string_value="vva_admin",
            description="Database username for IAM auth",
        )

        # ── Exports ─────────────────────────────────────────────────────

        self.db_secret = self.db_instance.secret

        CfnOutput(self, "RdsEndpoint",
                  value=self.db_instance.db_instance_endpoint_address)
        CfnOutput(self, "RdsProxyEndpoint",
                  value=self.rds_proxy.endpoint)
        CfnOutput(self, "DbSecretArn",
                  value=self.db_secret.secret_arn)
