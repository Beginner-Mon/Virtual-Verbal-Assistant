"""Kimodo ECS Stack — GPU-backed motion worker on EC2, driven by a DynamoDB queue.

Imports (pre-existing, created manually):
    - ASG: kimodo-asg (ECS GPU Optimized AMI ami-0a818a7573ba0a2ce)
    - IAM Roles: ecsTaskExecutionRoleKimodo, kimodo-task-role
    - Secrets Manager: ECA/kimodo-dtpy8O

Creates:
    - ECS Cluster (kimodo-cluster)
    - Security Group (ECS task, outbound only — see below)
    - Task Definition (4 vCPU, 12 GB RAM, 1 GPU, CloudWatch logs), running
      worker.py which polls the DynamoDB job queue instead of serving HTTP
    - DynamoDB table `vva-motion-jobs` (PAY_PER_REQUEST, TTL on `expires_at`,
      GSI `status-created_at-index` for the worker's poll query)
    - Gateway VPC endpoint for DynamoDB (free — unlike an interface endpoint,
      which bills hourly per AZ whether used or not)

This replaces an earlier design where the ECS task ran an HTTP/MCP server
behind an internet-facing ALB, so a Lambda agent could reach it inline. That
put a paid, always-on load balancer in front of a GPU box that is idle most
of the time. The worker now polls DynamoDB for queued jobs instead of
listening for inbound connections, so there is nothing left to load-balance:
no ALB, no target group, no listener, no ECS Service (the operator scales the
underlying ASG by hand — the task is a standalone poller, not a fronted
service).
"""

from aws_cdk import (
    RemovalPolicy,
    Stack,
    Tags,
    CfnOutput,
    aws_dynamodb as dynamodb,
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
        assets_bucket_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("Project", "ECA")
        Tags.of(self).add("Module", "Kimodo")

        # ── Security Group ─────────────────────────────────────────
        # The worker polls DynamoDB and S3 outbound; it never accepts an
        # inbound connection, so there is no port to open here at all.
        sg_ecs = ec2.SecurityGroup(
            self, "SgKimodoEcs",
            vpc=vpc,
            # Deliberately unnamed. A security group's Description is immutable
            # in CloudFormation, so editing it — as this branch did, dropping
            # "inbound 8000" for the queue worker — REPLACES the group. A fixed
            # security_group_name makes that replacement impossible, because
            # the new group is created before the old one is deleted:
            #   "Security Group with kimodo-sg-ecs already exists"
            # which is a rolled-back deploy, already paid for once. The two
            # settings together would make every future description edit
            # un-deployable. Nothing outside this stack referenced the name;
            # `aws ecs run-task` takes the id, published as a stack output.
            #
            # BUT REPLACING THIS GROUP BREAKS THE ASG, and nothing in CDK will
            # tell you. kimodo-asg is manual infrastructure (see the module
            # docstring's "Imports (pre-existing, created manually)") and its
            # launch template `kimodo-mcp-template` pins the group by ID. When
            # the replacement deleted sg-0dfc71f3914a3bc7b, every launch failed
            # with "The security group ... does not exist in VPC", visible only
            # in describe-scaling-activities — the ASG reports desired=1 and
            # zero instances, and CloudFormation is perfectly happy.
            #
            # So: after any deploy that replaces this group, publish a new
            # launch template version with the new id, which the ASG picks up
            # because it tracks $Latest:
            #   aws ec2 create-launch-template-version \
            #     --launch-template-id lt-0136cab703766c902 --source-version N \
            #     --launch-template-data '{"SecurityGroupIds":["<new sg>"]}'
            # The real fix is to bring the ASG into CDK so the reference is a
            # construct instead of a copied id; until then this comment is the
            # only thing linking the two.
            description="Kimodo motion worker - outbound only, no inbound listener",
            allow_all_outbound=True,
        )

        # ── Motion job table ────────────────────────────────────────
        self.motion_table = dynamodb.Table(
            self, "MotionJobs",
            table_name="vva-motion-jobs",
            partition_key=dynamodb.Attribute(
                name="job_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_at",
            removal_policy=RemovalPolicy.DESTROY,  # jobs are transient, TTL 24h
        )
        self.motion_table.add_global_secondary_index(
            index_name="status-created_at-index",
            partition_key=dynamodb.Attribute(
                name="status", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(
                name="created_at", type=dynamodb.AttributeType.NUMBER),
        )

        # Gateway endpoint — FREE. An interface endpoint bills hourly per AZ
        # whether used or not, which is the same disease as the ALB this stack
        # used to run. S3 and DynamoDB are the only two services with a gateway
        # endpoint, which is why the job queue lives on DynamoDB rather than SQS.
        #
        # Explicit L2 construct with scope=self (not vpc.add_gateway_endpoint):
        # that convenience method parents the endpoint under `vpc`, which was
        # constructed in VvaVpcStack — the resource would end up in the wrong
        # stack's template. Passing scope=self keeps it here, in KimodoEcsStack,
        # alongside the table it exists to reach.
        ec2.GatewayVpcEndpoint(
            self, "DynamoDbEndpoint",
            vpc=vpc,
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
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
        # No port_mappings, no health_check: worker.py polls DynamoDB, it does
        # not serve HTTP, so there is no port to expose and no endpoint to curl
        # (an HTTP health check here would just fail forever and get the task
        # killed).
        container = task_def.add_container(
            "kimodo-mcp",
            image=ecs.ContainerImage.from_registry(ECR_URI),
            command=["python", "worker.py"],
            gpu_count=1,
            essential=True,
            environment={
                "TEXT_ENCODER_MODE": "local",
                "TEXT_ENCODER_DEVICE": "cuda",
                "HF_HOME": "/workspace/.cache/huggingface",
                "MOTION_TABLE": self.motion_table.table_name,
                "MOTION_BUCKET": assets_bucket_name,
            },
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="mcp",
                log_group=log_group,
            ),
            linux_parameters=ecs.LinuxParameters(
                self, "LinuxParams",
                shared_memory_size=16384,
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

        # ── Task role grants ────────────────────────────────────────
        self.motion_table.grant_read_write_data(task_role)
        task_role.add_to_principal_policy(iam.PolicyStatement(
            actions=["s3:PutObject"],
            # Scoped to motions/* only — the worker writes rendered clips there,
            # it has no business touching the rest of the assets bucket.
            resources=[f"arn:aws:s3:::{assets_bucket_name}/motions/*"],
        ))

        # ── Outputs ────────────────────────────────────────────────
        #
        # There is no ECS Service, so nothing starts the worker on its own:
        # scaling kimodo-asg to 1 supplies an instance and nothing more. The
        # operator runs the task, and `aws ecs run-task --network-configuration`
        # needs both of these by id (network_mode is awsvpc). They are outputs
        # rather than names to retype — the group is deliberately unnamed, see
        # SgKimodoEcs above.
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
        CfnOutput(self, "TaskSecurityGroupId", value=sg_ecs.security_group_id)
        CfnOutput(self, "TaskDefinitionFamily", value=task_def.family)
