"""Kimodo ECS Stack — GPU-backed MCP server on EC2.

Imports (pre-existing, created manually):
    - ASG: kimodo-asg (ECS GPU Optimized AMI ami-0a818a7573ba0a2ce)
    - IAM Roles: ecsTaskExecutionRoleKimodo, kimodo-task-role
    - Secrets Manager: ECA/kimodo-dtpy8O

Creates:
    - ECS Cluster (kimodo-cluster)
    - Security Groups (ALB :80, ECS task :8000)
    - Task Definition (4 vCPU, 12 GB RAM, 1 GPU, CloudWatch logs)
    - When ENABLE_ALB=True:
        - ALB (internet-facing, :80, idle_timeout=300s for SSE)
        - Target Group (ip target, :8000, health check /mcp matcher 200-499)
        - ECS Service (CP strategy kimodo-GPU)

Toggle ENABLE_ALB=False + `cdk deploy` để teardown ALB/service.
"""

from aws_cdk import (
    Duration,
    Stack,
    Tags,
    CfnOutput,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_elasticloadbalancingv2 as elbv2,
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

ENABLE_ALB = True
# WARNING: MCP endpoint không có auth — đổi thành "<IP của bạn>/32" trước khi deploy.
ALB_ALLOWED_CIDR = "0.0.0.0/0"
VPC_CIDR = "10.0.0.0/16"


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

        # ── Security Groups ────────────────────────────────────────
        sg_alb = None
        if ENABLE_ALB:
            sg_alb = ec2.SecurityGroup(
                self, "SgKimodoAlb",
                vpc=vpc,
                security_group_name="kimodo-sg-alb",
                description="Kimodo ALB - inbound 80",
                allow_all_outbound=True,
            )
            sg_alb.add_ingress_rule(
                peer=ec2.Peer.ipv4(ALB_ALLOWED_CIDR),
                connection=ec2.Port.tcp(80),
                description="HTTP test endpoint",
            )

        sg_ecs = ec2.SecurityGroup(
            self, "SgKimodoEcs",
            vpc=vpc,
            security_group_name="kimodo-sg-ecs",
            description="Kimodo MCP - inbound 8000",
            allow_all_outbound=True,
        )
        if ENABLE_ALB:
            sg_ecs.add_ingress_rule(
                peer=ec2.Peer.security_group_id(sg_alb.security_group_id),
                connection=ec2.Port.tcp(8000),
                description="MCP via ALB",
            )
            sg_ecs.add_ingress_rule(
                peer=ec2.Peer.ipv4(VPC_CIDR),
                connection=ec2.Port.tcp(8000),
                description="MCP internal (SSM port-forward, VPC)",
            )
        else:
            sg_ecs.add_ingress_rule(
                peer=ec2.Peer.any_ipv4(),
                connection=ec2.Port.tcp(8000),
                description="MCP HTTP/SSE (direct task IP)",
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
            cpu="4096",
            memory_mib="12288",
        )

        # smplx-vol đã bỏ: kimodo giờ cài dạng wheel nên assets nằm trong site-packages,
        # không còn ở /workspace/kimodo — mount host volume đè lên một đường dẫn bên
        # trong package là chuyện dễ vỡ. smplx22 chỉ 12 KB và entrypoint vẫn sync từ S3
        # mỗi lần khởi động, nên S3 là nguồn sự thật duy nhất.
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
                "MCP_TTL_SECONDS": "3600",
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
                # FastMCP streamable-http trả 406 cho GET thuần → không dùng -f;
                # bất kỳ HTTP response nào cũng chứng minh server đang listen.
                command=["CMD-SHELL", "curl -sS -o /dev/null http://localhost:8000/mcp || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(300),
            ),
        )

        container.add_mount_points(
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

        # ── ALB + ECS Service ──────────────────────────────────────
        if ENABLE_ALB:
            alb = elbv2.ApplicationLoadBalancer(
                self, "KimodoAlb",
                load_balancer_name="kimodo-alb",
                vpc=vpc,
                internet_facing=True,
                vpc_subnets=ec2.SubnetSelection(subnet_group_name="EcsPublic"),
                security_group=sg_alb,
            )
            # SSE stream dài — idle timeout default 60s sẽ giết connection.
            alb.set_attribute("idle_timeout.timeout_seconds", "300")

            tg = elbv2.ApplicationTargetGroup(
                self, "KimodoTg",
                target_group_name="kimodo-mcp-tg",
                vpc=vpc,
                port=8000,
                protocol=elbv2.ApplicationProtocol.HTTP,
                target_type=elbv2.TargetType.IP,  # bắt buộc cho awsvpc
                health_check=elbv2.HealthCheck(
                    path="/mcp",
                    healthy_http_codes="200-499",  # 406 từ FastMCP vẫn healthy
                    interval=Duration.seconds(30),
                    timeout=Duration.seconds(10),
                    healthy_threshold_count=2,
                    unhealthy_threshold_count=3,
                ),
            )
            tg.set_attribute("deregistration_delay.timeout_seconds", "30")

            listener = alb.add_listener(
                "KimodoHttpListener",
                port=80,
                open=False,
                default_action=elbv2.ListenerAction.forward([tg]),
            )

            # L1 CfnService — bypass synth-validation hasEc2Capacity
            ecs_subnets = vpc.select_subnets(
                subnet_group_name="EcsPublic"
            )
            service = ecs.CfnService(
                self, "KimodoService",
                cluster=cluster.cluster_arn,
                task_definition=task_def.task_definition_arn,
                service_name="kimodo-mcp-service",
                desired_count=1,
                capacity_provider_strategy=[
                    ecs.CfnService.CapacityProviderStrategyItemProperty(
                        capacity_provider="kimodo-GPU",
                        weight=1,
                        base=0,
                    ),
                ],
                network_configuration=ecs.CfnService.NetworkConfigurationProperty(
                    awsvpc_configuration=ecs.CfnService.AwsVpcConfigurationProperty(
                        subnets=[s.subnet_id for s in ecs_subnets.subnets],
                        security_groups=[sg_ecs.security_group_id],
                        assign_public_ip="DISABLED",
                    ),
                ),
                load_balancers=[
                    ecs.CfnService.LoadBalancerProperty(
                        container_name="kimodo-mcp",
                        container_port=8000,
                        target_group_arn=tg.target_group_arn,
                    ),
                ],
                deployment_configuration=ecs.CfnService.DeploymentConfigurationProperty(
                    minimum_healthy_percent=1,
                    maximum_percent=200,
                    deployment_circuit_breaker=ecs.CfnService.DeploymentCircuitBreakerProperty(
                        enable=True,
                        rollback=True,
                    ),
                ),
                health_check_grace_period_seconds=600,
            )
            service.node.add_dependency(listener)

            CfnOutput(self, "ServiceName", value=service.service_name)
            CfnOutput(self, "AlbDnsName", value=alb.load_balancer_dns_name)

        # ── Outputs ────────────────────────────────────────────────
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
