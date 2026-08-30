# Kimodo on ECS (ECR + EC2-backed ECS) — Setup Guide

## Architecture

```
┌─────────────────────────────────────────────────┐
│ ECS Task (g5.xlarge)                            │
│                                                 │
│  ┌──────────────┐    ┌──────────────────┐       │
│  │ text-encoder │ ←→ │      demo        │       │
│  │  port 9550   │    │    port 7860     │       │
│  │  (Gradio)    │    │  (Gradio web UI) │       │
│  └──────────────┘    └────────┬─────────┘       │
│                               │                 │
│  NVIDIA A10G (24GB) ◄────────┘                 │
└─────────────────────────────────────────────────┘
         │
         ▼
    ALB (port 80 → 7860)
         │
         ▼
    User browser
```

Both containers share the same GPU (in one task). Text encoder runs Llama 8B (~16GB VRAM), demo runs Kimodo diffusion (~3GB VRAM). Total: ~19GB — fits on A10G 24GB.

---

## Prerequisites

Same as EC2 guide, plus:
- AWS CLI configured (`aws configure`)
- IAM permissions: ECR (read/write), ECS (task management), Secrets Manager

---

## 1. Create ECR Repository

```bash
aws ecr create-repository --repository-name kimodo
# Note the repository URI:
# 123456789.dkr.ecr.us-east-1.amazonaws.com/kimodo
```

---

## 2. Build, Tag, and Push Docker Image

> **Outdated instructions removed.** This section used to say `docker compose build` and
> push the result as `kimodo:latest`. That builds the **dev** image (`Dockerfile`, NGC
> PyTorch base, gradio + kimodo-viser), which is not what production runs.

Production uses **two** images, built by CI, never by hand:

| Image | Dockerfile | Contains | Rebuilt when |
|---|---|---|---|
| `kimodo-base:vN` | `Dockerfile.base` | Third-party deps from the committed lockfile + compiled `motion_correction` | deps / `MotionCorrection/**` / `scripts/**` / `Dockerfile.base` change |
| `kimodo:latest` | `Dockerfile.prod` | `FROM kimodo-base:vN` + the kimodo wheel + `mcp_server.py` | any push to `text-to-motion/kimodo/**` |

Both are pushed by GitHub Actions on a push to the `kimodo-release` branch —
`.github/workflows/build-kimodo-base.yml` and `.github/workflows/deploy-production.yml`.
The prod build takes ~30 seconds because it only builds a pure-Python wheel; the base
build takes ~15 minutes and is the one that downloads torch.

**Ordering rule:** `Dockerfile.prod` starts `FROM kimodo-base:${BASE_TAG}`, so bumping
`BASE_TAG` requires the new base tag to exist in ECR *first*. Dispatch
`build-kimodo-base.yml` (with the new `base_tag`) and let it finish before merging the
change that flips `BASE_TAG` in `deploy-production.yml` — otherwise the prod job races a
base image that is not there yet and fails on the `FROM` line.

To regenerate the production lockfile after editing `docker_requirements_prod.in`:

```bash
cd text-to-motion/kimodo
python kimodo/scripts/lock_requirements.py --prod   # writes docker_requirements_prod.txt
```

Commit that lockfile. `Dockerfile.base` installs it with `--no-deps`, so it is the sole
authority on what lands in the image — nothing is resolved at build time.

---

## 3. Store HuggingFace Token in Secrets Manager

```bash
aws secretsmanager create-secret \
  --name kimodo-hf-token \
  --secret-string '{"HF_TOKEN":"hf_your_token_here"}'
```

---

## 4. Create ECS Cluster (EC2-backed)

```bash
# Create cluster
aws ecs create-cluster --cluster-name kimodo-cluster

# Create launch template for g5.xlarge
aws ec2 create-launch-template \
  --launch-template-name kimodo-gpu \
  --launch-template-data '{
    "ImageId": "ami-0d1d4da6c8a72b446",
    "InstanceType": "g5.xlarge",
    "KeyName": "YOUR_KEY_NAME",
    "BlockDeviceMappings": [{
      "DeviceName": "/dev/xvda",
      "Ebs": {"VolumeSize": 150, "VolumeType": "gp3"}
    }],
    "UserData": "IyEvYmluL2Jhc2gKZWNobyAiRUNTX0NMVVNURVI9a2ltb2RvLWNsdXN0ZXIiID4+IC9ldGMvZWNzL2Vjcy5jb25maWc="
  }'
```

The UserData (base64-decoded) is:
```bash
#!/bin/bash
echo "ECS_CLUSTER=kimodo-cluster" >> /etc/ecs/ecs.config
```

### Create Capacity Provider + Auto Scaling Group

```bash
# Create ASG
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name kimodo-asg \
  --launch-template LaunchTemplateName=kimodo-gpu \
  --min-size 1 --max-size 1 --desired-capacity 1 \
  --vpc-zone-identifier "subnet-xxx,subnet-yyy"

# Create capacity provider
aws ecs create-capacity-provider \
  --name kimodo-capacity \
  --auto-scaling-group-provider "autoScalingGroupArn=arn:aws:autoscaling:...:autoScalingGroup:...:autoScalingGroupName/kimodo-asg,managedScaling={status=ENABLED,targetCapacity=100}"
```

> **Simpler alternative**: Use AWS Console → ECS → Create Cluster → EC2 Linux → g5.xlarge

---

## 5. Create ECS Task Definition

Create a JSON file `task-definition.json`:

```json
{
  "family": "kimodo",
  "networkMode": "awsvpc",
  "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789:role/ecsTaskRole",
  "cpu": "4096",
  "memory": "16384",
  "requiresCompatibilities": ["EC2"],
  "containerDefinitions": [
    {
      "name": "text-encoder",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/kimodo:latest",
      "command": ["python", "-m", "kimodo.scripts.run_text_encoder_server"],
      "essential": true,
      "portMappings": [{"containerPort": 9550, "protocol": "tcp"}],
      "environment": [
        {"name": "HF_HOME", "value": "/workspace/.cache/huggingface"},
        {"name": "GRADIO_SERVER_NAME", "value": "0.0.0.0"},
        {"name": "GRADIO_SERVER_PORT", "value": "9550"}
      ],
      "secrets": [
        {"name": "HF_TOKEN", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:kimodo-hf-token:HF_TOKEN::"}
      ],
      "resourceRequirements": [{"type": "GPU", "value": "1"}],
      "linuxParameters": {"sharedMemorySize": 16384},
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:9550/ || exit 1"],
        "interval": 10,
        "timeout": 5,
        "retries": 60,
        "startPeriod": 120
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/kimodo",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "text-encoder"
        }
      }
    },
    {
      "name": "demo",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/kimodo:latest",
      "command": ["python", "-m", "kimodo.demo"],
      "essential": true,
      "portMappings": [{"containerPort": 7860, "protocol": "tcp"}],
      "dependsOn": [
        {"containerName": "text-encoder", "condition": "HEALTHY"}
      ],
      "environment": [
        {"name": "TEXT_ENCODER_URL", "value": "http://localhost:9550/"},
        {"name": "SERVER_PORT", "value": "7860"},
        {"name": "HF_HOME", "value": "/workspace/.cache/huggingface"}
      ],
      "secrets": [
        {"name": "HF_TOKEN", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:kimodo-hf-token:HF_TOKEN::"}
      ],
      "resourceRequirements": [{"type": "GPU", "value": "1"}],
      "linuxParameters": {"sharedMemorySize": 16384},
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:7860/ || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/kimodo",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "demo"
        }
      }
    }
  ]
}
```

Register the task definition:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Important Notes on the Task Definition

- **Both containers get GPU**: `"resourceRequirements": [{"type": "GPU", "value": "1"}]` — ECS shares the same GPU across containers in the same task.
- **`dependsOn`**: The demo waits for text-encoder to be healthy before starting.
- **`localhost`**: Containers in the same task share the network namespace — the demo connects to text-encoder via `http://localhost:9550/`.
- **`shm_size`**: Set to 16GB via `linuxParameters.sharedMemorySize` for large model loading.
- **No volume mounts needed**: HF models download directly inside the container. The HF token comes from Secrets Manager.

---

## 6. Create Application Load Balancer (ALB)

The ALB routes to the demo's port 7860.

```bash
# Create target group
aws elbv2 create-target-group \
  --name kimodo-tg \
  --protocol HTTP --port 7860 \
  --target-type ip \
  --vpc-id vpc-xxxxxxxx \
  --health-check-path / \
  --health-check-interval-seconds 60 \
  --health-check-timeout-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 5

# Create ALB
aws elbv2 create-load-balancer \
  --name kimodo-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxxxxxxx \
  --scheme internet-facing

# Create listener (port 80 → target group)
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/kimodo-alb/... \
  --protocol HTTP --port 80 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/kimodo-tg/...
```

> **Simpler**: Create the ALB in AWS Console → assign to the ECS service.

---

## 7. Create ECS Service

```bash
aws ecs create-service \
  --cluster kimodo-cluster \
  --service-name kimodo-service \
  --task-definition kimodo \
  --desired-count 1 \
  --launch-type EC2 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxxxxxxx],assignPublicIp=ENABLED}" \
  --load-balancers "containerName=demo,containerPort=7860,targetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/kimodo-tg/..."
```

> **Or**: Use AWS Console → ECS → Create Service → EC2 → pick the task definition

---

## 8. Access the Demo

```
http://<ALB_DNS_NAME>/
```

Find the ALB DNS name:
```bash
aws elbv2 describe-load-balancers --names kimodo-alb --query 'LoadBalancers[0].DNSName'
```

---

## 9. IAM Roles Summary

| Role | Permissions |
|------|------------|
| `ecsTaskExecutionRole` | ECR pull, CloudWatch logs, Secrets Manager read |
| `ecsTaskRole` | Same as above + S3 (if exporting motions) |

Minimal trust policy for both:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

---

## 10. Cleanup

```bash
aws ecs delete-service --cluster kimodo-cluster --service kimodo-service --force
aws ecs delete-cluster --cluster kimodo-cluster
aws ecr delete-repository --repository-name kimodo --force
aws secretsmanager delete-secret --secret-id kimodo-hf-token --force-delete-without-recovery
# Terminate EC2 instance via ASG or Console
```

---

## Full Deployment Script (One-Shot)

Save as `deploy.sh`:

```bash
#!/bin/bash
set -e

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/kimodo"
CLUSTER="kimodo-cluster"

echo "=== 1. Build and push to ECR ==="
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ECR_URI
docker tag kimodo:1.0 $ECR_URI:latest
docker push $ECR_URI:latest

echo "=== 2. Register task definition ==="
aws ecs register-task-definition --cli-input-json file://task-definition.json

echo "=== 3. Update service ==="
aws ecs update-service \
  --cluster $CLUSTER \
  --service kimodo-service \
  --task-definition kimodo \
  --force-new-deployment

echo "=== Done ==="
echo "ALB: $(aws elbv2 describe-load-balancers --names kimodo-alb --query 'LoadBalancers[0].DNSName' --output text)"
```

---

## Cost Estimate

| Resource | Monthly (spot) | Monthly (on-demand) |
|----------|---------------|---------------------|
| g5.xlarge (ECS) | ~$250 | ~$750 |
| ALB | ~$20 | ~$20 |
| EBS 150GB gp3 | ~$12 | ~$12 |
| ECR (40GB) | ~$4 | ~$4 |
| **Total** | **~$285/mo** | **~$785/mo** |

Per-hour: ~$0.35 spot vs ~$1.00 on-demand. Stop the service when not in use.
