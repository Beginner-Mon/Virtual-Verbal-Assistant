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

## SSM parameters and deploy-time flags

Every value here is either a secret (so it must not enter a CloudFormation
template) or an identifier that does not exist until something else has been
deployed. None of them have usable defaults, and a stack that is missing one
now **fails at synth** rather than deploying something broken.

### The parameters, once per account

```bash
# Neon, pooled endpoint — read by VvaCrudApiStack and VvaAgentStack.
# POOLED here (with "-pooler"), unlike /vva/neon/dsn above, which is direct.
aws ssm put-parameter --name /vva/neon/dsn-pooler --type SecureString \
  --value 'postgresql://USER:PASS@ep-xxx-pooler.c-NN.us-east-1.aws.neon.tech/neondb?sslmode=require'

# LLM credentials. Lambda environment variables are plaintext in the template
# and CloudFormation's {{resolve:ssm-secure}} is not supported for them, so
# llm.py reads these at run time instead.
aws ssm put-parameter --name /vva/llm/deepseek-api-key --type SecureString --value 'sk-...'
aws ssm put-parameter --name /vva/llm/gemini-api-keys  --type SecureString --value 'key1,key2'

# The CloudFront PRIVATE signing key for motions/* (see below for generating it).
aws ssm put-parameter --name /vva/motion/signing-key-pem --type SecureString \
  --value "$(cat motion_signing_key.pem)"

# HMAC secret for the motion job id. Not the signing key and not interchangeable
# with it: this one makes the DynamoDB key underivable from the prompt while
# keeping identical prompts on the same key, so the cache still hits.
# Any 32+ random bytes. CHANGING IT INVALIDATES EVERY CACHED MOTION — the job id
# is the hash, so every prompt re-renders on the GPU once.
aws ssm put-parameter --name /vva/motion/hash-secret --type SecureString \
  --value "$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

| Parameter | Read by | Held as |
|---|---|---|
| `/vva/neon/dsn` | `VvaCharacterStack` | SecureString, DIRECT endpoint |
| `/vva/neon/dsn-pooler` | `VvaCrudApiStack`, `VvaAgentStack` | SecureString, POOLED endpoint |
| `/vva/llm/deepseek-api-key` | `VvaAgentStack` (`llm.py`) | SecureString |
| `/vva/llm/gemini-api-keys` | `VvaAgentStack` (`llm.py`) | SecureString, comma-separated |
| `/vva/motion/signing-key-pem` | `VvaAgentStack` (`api/motion_status.py`) | SecureString |
| `/vva/motion/hash-secret` | `VvaAgentStack` (`nodes/kimodo.py`) | SecureString |

The stacks are given the parameter **NAME**, never the value — the Lambda
resolves it at call time with `ssm:GetParameter(WithDecryption=True)`. No secret
ever reaches a CDK context flag or the synthesized template, and
`tests/infra/test_motion_route_infra.py` asserts exactly that.

### The CloudFront signing keypair for motions/*

`motions/*` is served only through signed URLs, so it needs an RSA keypair whose
halves are deployed to two different places. Generate it once and keep the
private half out of the repo:

```bash
openssl genrsa -out motion_signing_key.pem 2048
openssl rsa -in motion_signing_key.pem -pubout -out motion_signing_key.pub
```

* **public half** → a `-c` flag on `VvaAssetStack`, which registers it as a
  CloudFront public key. It is not read from SSM: CDK needs it at synth time to
  build the key group.
* **private half** → the `/vva/motion/signing-key-pem` SecureString above. It
  never appears in CDK at all.

### Deploy order — and why it is a strict order

`motion_key_pair_id` is assigned **by CloudFront**, so it does not exist until
`VvaAssetStack` has been deployed once. `VvaAgentStack` cannot be deployed
correctly before that.

```bash
# 1. Register the public key. Fails at synth without the flag.
cdk deploy VvaAssetStack -c motion_public_key_pem="$(cat motion_signing_key.pub)"

# 2. Read back the id CloudFront just assigned.
aws cloudfront list-public-keys \
  --query 'PublicKeyList.Items[].{Id:Id,Name:Name}' --output table

# 3. Agent, two-step the first time (repository, then function).
cdk deploy VvaAgentStack -c agent_bootstrap=1
#    ... CI pushes vva-agent:<sha> (deploy-agent.yml) ...
cdk deploy VvaAgentStack -c agent_image_tag=<sha> -c motion_key_pair_id=<K2EXAMPLE...>
```

Step 3 without `motion_key_pair_id` used to synthesize and deploy cleanly, then
hand out signed URLs CloudFront answers with **403** — the environment variable
was present, merely empty, so nothing raised anywhere. Same for the CDN origin
(`asset_base_url`, passed from `VvaAssetStack` by `app.py`), whose absence
produced URLs with no host. Both are now `Annotations.add_error` at synth,
scoped to `VvaAgentStack` so unrelated stacks still synthesize.

### Schema

The Alembic revisions under
`agenticRAG/langgraph_agents/alembic/versions/` are the only schema system in
this repo with a runner. `infra/sql/init_schema.sql` is a stale reference copy —
nothing executes it, and it has drifted (`tokens` where the code reads
`token_count`). Run migrations before deploying an agent image that depends on
a new column:

```bash
cd agenticRAG/langgraph_agents && alembic upgrade head
```

It needs `VVA_PG_DSN_OWNER` (falling back to `VVA_PG_DSN`) — the DIRECT
endpoint, and the credential that OWNS the tables, not `eca_user`. Since
007_rls the application connects as `eca_user`, which has no CREATE and is
subject to row-level security, so `alembic upgrade` as that role fails on the
first DDL statement. `env.py` gives the owner credential its own variable name
for exactly that reason.

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
