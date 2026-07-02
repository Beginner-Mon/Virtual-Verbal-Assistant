"""Kimodo ECS Stack — GPU-backed MCP server on EC2.

Imports (pre-existing, created manually):
    - ECS Cluster: kimodo-cluster
    - ASG: kimodo-asg (ECS GPU Optimized AMI ami-0a818a7573ba0a2ce)
    - IAM Roles: ecsTaskExecutionRoleKimodo, kimodo-task-role
    - Secrets Manager: ECA/kimodo-dtpy8O

Creates:
    - Security Group (port 8000)
    - Task Definition (1 vCPU, 4 GB RAM, 1 GPU, CloudWatch logs)
    - ECS Service (EC2 direct launch, EcsPublic subnets, public IP)
"""

from aws_cdk import (
    Duration,
    Stack,
    Tags,
    CfnOutput,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

ACCOUNT = "244203483654"
REGION = "us-east-1"
ECR_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/kimodo:latest"
SECRET_ARN = f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:ECA/kimodo-dtpy8O"

EXEC_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/ecsTaskExecutionRoleKimodo"
TASK_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/kimodo-task-role"


class KimodoEcsStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.Vpc,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("Project", "ECA")
        Tags.of(self).add("Module", "Kimodo")

        # ── Security Group ─────────────────────────────────────────
        sg_ecs = ec2.SecurityGroup(
            self, "SgKimodoEcs",
            vpc=vpc,
            security_group_name="kimodo-sg-ecs",
            description="Kimodo MCP - inbound 8000",
            allow_all_outbound=True,
        )
        sg_ecs.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(8000),
            description="MCP HTTP/SSE",
        )

        # ── CloudWatch Log Group ───────────────────────────────────
        log_group = logs.LogGroup(
            self, "KimodoLogGroup",
            log_group_name="/ecs/kimodo",
            retention=logs.RetentionDays.ONE_WEEK,
        )

        # ── Create ECS Cluster ─────────────────────────────────────
        cluster = ecs.Cluster(
            self, "KimodoCluster",
            cluster_name="kimodo-cluster",
            vpc=vpc,
            container_insights=True,
        )

        execution_role = iam.Role.from_role_arn(
            self, "ExecRole",
            role_arn=EXEC_ROLE_ARN,
        )
        task_role = iam.Role.from_role_arn(
            self, "TaskRole",
            role_arn=TASK_ROLE_ARN,
        )

        # ── Task Definition ────────────────────────────────────────
        task_def = ecs.TaskDefinition(
            self, "KimodoTaskDef",
            family="kimodo-mcp",
            compatibility=ecs.Compatibility.EC2,
            network_mode=ecs.NetworkMode.AWS_VPC,
            execution_role=execution_role,
            task_role=task_role,
            cpu="1024",
            memory_mib="4096",
        )

        task_def.add_volume(
            name="smplx-vol",
            host=ecs.Host(source_path="/root/kimodo/assets/skeletons/smplx22"),
        )
        task_def.add_volume(
            name="hf-cache-vol",
            host=ecs.Host(source_path="/mnt/instance-store/huggingface"),
        )

        # ── Container ──────────────────────────────────────────────
        container = task_def.add_container(
            "kimodo-mcp",
            image=ecs.ContainerImage.from_registry(ECR_URI),
            command=["python", "mcp_server.py"],
            gpu_count=1,
            essential=True,
            port_mappings=[
                ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP),
            ],
            environment={
                "TEXT_ENCODER_MODE": "local",
                "TEXT_ENCODER_DEVICE": "cuda",
                "HF_HOME": "/workspace/.cache/huggingface",
                "MCP_PORT": "8000",
            },
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="mcp",
                log_group=log_group,
            ),
            linux_parameters=ecs.LinuxParameters(
                self, "LinuxParams",
                shared_memory_size=16384,
            ),
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -fsS http://localhost:8000/mcp || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(300),
            ),
        )

        container.add_mount_points(
            ecs.MountPoint(
                container_path="/workspace/kimodo/assets/skeletons/smplx22",
                source_volume="smplx-vol",
                read_only=False,
            ),
            ecs.MountPoint(
                container_path="/workspace/.cache/huggingface",
                source_volume="hf-cache-vol",
                read_only=False,
            ),
        )

        hf_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "HfTokenSecret",
            secret_complete_arn=SECRET_ARN,
        )
        container.add_secret(
            "HF_TOKEN",
            ecs.Secret.from_secrets_manager(hf_secret, field="HF_TOKEN"),
        )

        # ── ECS Service (Commented out per user request) ───────────
        # service = ecs.Ec2Service(
        #     self, "KimodoService",
        #     cluster=cluster,
        #     task_definition=task_def,
        #     service_name="kimodo-mcp-service",
        #     desired_count=1,
        #     vpc_subnets=ec2.SubnetSelection(subnet_group_name="EcsPublic"),
        #     security_groups=[sg_ecs],
        #     assign_public_ip=True,
        #     min_healthy_percent=0,
        #     max_healthy_percent=100,
        #     circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        # )

        # ── Outputs ────────────────────────────────────────────────
        # CfnOutput(self, "ServiceName", value=service.service_name)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
