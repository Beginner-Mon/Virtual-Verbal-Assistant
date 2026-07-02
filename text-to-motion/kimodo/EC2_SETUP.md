# Kimodo MCP Server — EC2 Setup Guide (Tested)

> This guide was tested on a **Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)** instance. All commands are copy-paste ready.

---

## Prerequisites

| Item | Details |
|------|---------|
| **AWS Account** | With GPU instance quota (see Step 0) |
| **EC2 AMI** | Deep Learning Base AMI with Single CUDA (Amazon Linux 2023), **64-bit (x86)** |
| **Instance Type** | `g5.xlarge` (recommended) or `g4dn.xlarge` (budget) or `t3.2xlarge` (CPU-only testing) |
| **EBS Storage** | **150 GB**, type **gp3** |
| **Security Group** | Allow ports: **22** (SSH), **8000** (MCP server) |
| **Docker Image** | Already pushed to ECR at `244203483654.dkr.ecr.us-east-1.amazonaws.com/kimodo:latest` |
| **HuggingFace** | Account with access granted to [nvidia/Kimodo-SMPLX-RP-v1](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) |
| **SMPL-X Model** | `SMPLX_NEUTRAL.npz` downloaded from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) |

---

## Step 0: Request GPU Instance Quota (One-Time)

New AWS accounts have a **0 vCPU limit** for GPU instances. You must request an increase before launching a GPU instance.

1. Go to **AWS Console → Service Quotas → Amazon EC2**
2. Search for `Running On-Demand G and VT instances` (or `All G and VT Spot Instance Requests` for spot)
3. Click **Request quota increase** → enter **4** → submit
4. Wait for approval (minutes to hours)

> **Note**: If you just want to test the plumbing without a GPU, you can skip this and use a `t3.2xlarge` CPU instance instead. Generation will be very slow (~5-15 min per request) but the API will work.

---

## Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Select AMI: **Deep Learning Base AMI with Single CUDA (Amazon Linux 2023)** — `64-bit (x86)`
3. Instance type: `g5.xlarge` (or `g4dn.xlarge` / `t3.2xlarge`)
4. Storage: **150 GiB**, **gp3**
5. Security group: Allow **TCP 22** and **TCP 8000**
6. Launch with your key pair

---

## Step 2: Upload SMPLX_NEUTRAL.npz to EC2

From your **local PC** (PowerShell/terminal), upload the SMPL-X body model file:

```bash
scp -i YOUR_KEY.pem SMPLX_NEUTRAL.npz ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/SMPLX_NEUTRAL.npz
```

---

## Step 3: SSH into EC2

```bash
ssh -i YOUR_KEY.pem ec2-user@<EC2_PUBLIC_IP>
```

Then switch to root (the Deep Learning AMI often requires root for Docker):

```bash
sudo su -
```

---

## Step 4: Start Docker

```bash
sudo systemctl start docker
sudo usermod -aG docker ec2-user
```

---

## Step 5: Configure AWS CLI

```bash
aws configure
```

Enter:
- **AWS Access Key ID**: *(your key)*
- **AWS Secret Access Key**: *(your secret)*
- **Default region name**: `us-east-1`
- **Default output format**: `json`

---

## Step 6: Create Directories

```bash
mkdir -p /root/.cache/huggingface
mkdir -p /root/kimodo/assets/skeletons/smplx22
```

---

## Step 7: Save HuggingFace Token

Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then write it directly to file:

```bash
echo -n "hf_YOUR_TOKEN_HERE" > /root/.cache/huggingface/token
```

Verify:
```bash
cat /root/.cache/huggingface/token
```

---

## Step 8: Authenticate Docker with ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 244203483654.dkr.ecr.us-east-1.amazonaws.com
```

Expected output: `Login Succeeded`

---

## Step 9: Copy Skeleton Files from Docker Image

The Docker image contains skeleton helper files (`joints.p`, `beta.npy`, `mean_hands.npy`) that must be extracted to the host before mounting the volume:

```bash
docker run --rm \
  -v /root/kimodo/assets/skeletons/smplx22:/host_skeleton \
  244203483654.dkr.ecr.us-east-1.amazonaws.com/kimodo:latest \
  bash -c "cp /workspace/kimodo/assets/skeletons/smplx22/* /host_skeleton/"
```

---

## Step 10: Copy SMPLX_NEUTRAL.npz to Skeleton Directory

```bash
cp /home/ec2-user/SMPLX_NEUTRAL.npz /root/kimodo/assets/skeletons/smplx22/
```

Verify all 4 files are present:
```bash
ls -la /root/kimodo/assets/skeletons/smplx22/
```

Expected:
```
beta.npy
joints.p
mean_hands.npy
SMPLX_NEUTRAL.npz
```

---

## Step 11: Run the MCP Server

### For GPU instances (g5.xlarge / g4dn.xlarge):
```bash
docker run -d \
  --name mcp-server \
  --gpus all \
  --shm-size=16gb \
  --network host \
  -e TEXT_ENCODER_MODE=local \
  -e TEXT_ENCODER_DEVICE=cpu \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v /root/.cache/huggingface:/workspace/.cache/huggingface \
  -v /root/kimodo/assets/skeletons/smplx22:/workspace/kimodo/assets/skeletons/smplx22:ro \
  244203483654.dkr.ecr.us-east-1.amazonaws.com/kimodo:latest python mcp_server.py
```

### For CPU-only instances (t3.2xlarge):
```bash
docker run -d \
  --name mcp-server \
  --shm-size=16gb \
  --network host \
  -e TEXT_ENCODER_MODE=local \
  -e TEXT_ENCODER_DEVICE=cpu \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v /root/.cache/huggingface:/workspace/.cache/huggingface \
  -v /root/kimodo/assets/skeletons/smplx22:/workspace/kimodo/assets/skeletons/smplx22:ro \
  244203483654.dkr.ecr.us-east-1.amazonaws.com/kimodo:latest python mcp_server.py
```

*(The only difference is `--gpus all` is removed for CPU instances.)*

---

## Step 12: Monitor Startup

```bash
docker logs -f mcp-server
```

Wait until you see: `Starting MCP server on port 8000...`

First startup downloads ~15-20 GB of AI models. This takes **5-10 minutes**.

Press `Ctrl+C` to exit log view (server keeps running).

---

## Step 13: Test the Server

```bash
# Health check
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"health_check","arguments":{}},"id":1}'

# List models
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_models","arguments":{}},"id":2}'

# Generate motion
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"generate_motion","arguments":{"prompt":"A person walks forward and waves"}},"id":3}'
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `vCPU limit of 0` | Request quota increase in Service Quotas (Step 0) |
| `Unable to locate credentials` | Run `aws configure` (Step 5) |
| `GatedRepoError / 401` | Check HF token: `cat /root/.cache/huggingface/token` and verify access at [model page](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) |
| `No such file: joints.p` | Run Step 9 to extract skeleton files from the Docker image |
| `Illegal header value 'Bearer '` | Token file is empty. Re-run Step 7 with your actual token |
| Container exits immediately | Check `docker logs mcp-server` for the error |
| Port 8000 not reachable externally | Check EC2 security group allows inbound TCP 8000 |

---

## Cleanup

```bash
# Stop and remove the container
docker rm -f mcp-server

# IMPORTANT: Terminate the EC2 instance from the AWS Console to stop ALL charges.
# Just "stopping" the instance still incurs EBS storage charges (~$12/month).
```
