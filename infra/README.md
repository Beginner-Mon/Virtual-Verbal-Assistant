# VVA Infrastructure (AWS CDK, Python)

Two architectures live in this directory. Only one of them is meant to be
deployed. Read this before running any `cdk` command.

> **Never run `cdk deploy --all`.** Name the stack. See "Track 1" below for why.

---

## Track 2 — cost-optimised (active)

```
Internet → CloudFront ─┬─ /characters*  → Lambda Function URL (OUTSIDE any VPC)
                       │                       ↓ TLS, same region
                       │                    Neon Postgres (us-east-1)
                       └─ /*             → S3 (private, OAC)
```

| Stack | File | What it is |
|---|---|---|
| `VvaCharacterStack` | [infra/character_stack.py](infra/character_stack.py) | `vva-characters` Lambda + Function URL (AWS_IAM). Reads the Neon DSN from SSM. |
| `VvaAssetStack` | [infra/asset_stack.py](infra/asset_stack.py) | Private S3 bucket + CloudFront. Serves `.vrm` models, routes `/characters*` to the Lambda. |
| `VvaVpcStack` | [infra/vpc_stack.py](infra/vpc_stack.py) | Deployed, costs ~nothing. `nat_gateways=0`, no interface endpoints, only the free S3 gateway endpoint. Kimodo ECS uses it. |
| `VvaKimodoEcsStack` | [infra/kimodo_ecs_stack.py](infra/kimodo_ecs_stack.py) | GPU MCP server. Awaiting Owner (cost). |

**The Lambda is deliberately outside the VPC.** Neon is a public TLS endpoint,
so putting the function in the private isolated subnets would need either a NAT
gateway (~$32/month) or interface VPC endpoints (~$7/month each, per AZ) just to
reach a service already on the internet — and would add ENI attachment to every
cold start. Outside the VPC it is cheaper *and* faster. With Neon, "put it in
the VPC to be safe" means paying to run slower.

The database is not hidden behind a VPC any more. What replaces that: TLS in
transit, a SecureString-held credential, and `AWS_IAM` auth on the Function URL
so only CloudFront's OAC can invoke it.

### Deploy

```bash
# Prerequisite, once — the DSN is a secret and must not enter a CFN template.
# Use the DIRECT endpoint: the pooled hostname with only "-pooler" removed.
# Newer Neon hostnames carry a ".c-NN" segment that must be KEPT — dropping it
# as well fails authentication.
aws ssm put-parameter --name /vva/neon/dsn --type SecureString \
  --value 'postgresql://USER:PASS@ep-xxx.c-NN.us-east-1.aws.neon.tech/neondb?sslmode=require'

cdk deploy VvaCharacterStack     # needs Docker running (layer bundling)
cdk deploy VvaAssetStack

# Then seed the catalog and upload the models:
python ../scripts/upload_characters_to_s3.py \
    --bucket "<AssetBucketName>" --cdn "https://<AssetBaseUrl>"
```

Feed `AssetBaseUrl` to the frontend as `VITE_ASSET_BASE_URL` and set
`VITE_USE_S3_MODELS=true`.

Useful context knobs:

```bash
cdk deploy VvaAssetStack -c allowed_origins=https://app.example.com,http://localhost:5173
cdk deploy VvaCharacterStack -c neon_dsn_param=/vva/neon/dsn-staging
```

---

## Track 1 — production reference (frozen, NOT synthesised)

```
Internet → API Gateway → Lambda (VPC private isolated)
                            ↓ IAM auth + TLS
                         RDS Proxy → RDS Postgres
```

| Stack | File |
|---|---|
| `VvaDbStack` | [infra/database_stack.py](infra/database_stack.py) |
| `VvaLambdaStack` | [infra/lambda_stack.py](infra/lambda_stack.py) |
| `VvaApiStack` | [infra/api_gateway_stack.py](infra/api_gateway_stack.py) |

The application moved to Neon on 31/07/2026 and these stacks have never been
deployed. They are kept as the architecture to return to when there is budget
for it: the database never reachable from the internet, IAM auth instead of a
static credential, RDS Proxy pooling connections.

They are **excluded from synthesis by default**, because `cdk deploy --all`
would otherwise stand up an RDS instance and an RDS Proxy that nothing uses.
To work on them:

```bash
CDK_ENABLE_TRACK1=1 cdk synth VvaDbStack
```

Reactivating the track is its own piece of work with its own costs — RDS
instance and storage, RDS Proxy per vCPU-hour, and the interface VPC endpoints
that were removed from `vpc_stack.py` precisely because they cost money while
nothing was running. Write a plan for it then; do not treat the code sitting
here as a to-do list.

---

## Lambda layer

Both tracks share [lambda/layer/shared/](lambda/layer/shared), which bundles
pg8000 plus `db.py` and `response.py`. `db.py` branches on `DB_MODE`:

| `DB_MODE` | Connection |
|---|---|
| `neon` | one SSM SecureString holding the DSN, split into pg8000's kwargs |
| `rds` (default) | four SSM parameters + an IAM auth token refreshed every ~14 min |

One module with two branches rather than two files, so the connection cache and
the JSON type serializers cannot drift apart.

Bundling runs `pip install` inside the Python 3.12 Lambda image, so **Docker
must be running** for any deploy that touches a layer.
