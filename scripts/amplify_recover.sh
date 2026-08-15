#!/usr/bin/env bash
#
# Recover a stuck Amplify Gen2 branch deployment, from the CLI.
#
# Everything the Amplify/CloudFormation consoles do for this is an API call, so
# none of it needs a browser. What it does need is AWS credentials — that, not
# tooling, is what blocks it.
#
# Terraform is not the answer here: Amplify Gen2 already synthesises CDK into
# CloudFormation, and a second IaC tool managing the same stack fights the first.
#
# Read-only by default. Every destructive step needs its own flag.
#
#   ./scripts/amplify_recover.sh status          # what state is the stack in?
#   ./scripts/amplify_recover.sh secrets         # which secrets exist, and where
#   ./scripts/amplify_recover.sh delete-stack    # DESTRUCTIVE, asks first
#   ./scripts/amplify_recover.sh set-env
#   ./scripts/amplify_recover.sh deploy
#
# Credentials come from the normal AWS chain (`aws configure`, SSO, or
# AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in the environment). Never paste keys
# into a chat window or into this file.
#
# Secret VALUES are read from the environment so they never land in shell
# history or a log:
#   GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=... ./scripts/amplify_recover.sh set-secrets

set -euo pipefail

APP_ID="${AMPLIFY_APP_ID:-d32nf9wwqqt016}"
BRANCH="${AMPLIFY_BRANCH:-feature/langgraph-rewrite}"
REGION="${AWS_REGION:-us-east-1}"

# Amplify flattens the branch name into the stack name; this is the one from the
# failing build log.
STACK="${AMPLIFY_STACK:-amplify-${APP_ID}-featurelanggraphrewrite-branch-98ebbaf816}"

aws() { command aws --region "$REGION" "$@"; }

require_aws() {
  # `type -P` looks only at PATH. `command -v` would find the wrapper function
  # defined above and report the CLI as installed when it is not — which turns a
  # missing install into a misleading "no usable credentials".
  if ! type -P aws >/dev/null 2>&1; then
    cat >&2 <<'MSG'
aws CLI not found. Install it, then re-run:

  winget install --id Amazon.AWSCLI -e        # Windows
  brew install awscli                         # macOS

Then authenticate — do NOT paste keys into a chat:
  aws configure                               # or: aws sso login --profile <name>
MSG
    exit 127
  fi
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "aws CLI found but no usable credentials. Run 'aws configure' first." >&2
    exit 1
  fi
  echo "account: $(aws sts get-caller-identity --query Account --output text)  region: $REGION"
}

cmd_status() {
  require_aws
  echo
  echo "== CloudFormation stack =="
  local status
  status=$(aws cloudformation describe-stacks --stack-name "$STACK" \
             --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND")
  echo "  $STACK"
  echo "  status: $status"

  case "$status" in
    ROLLBACK_COMPLETE|CREATE_FAILED)
      echo
      echo "  The CREATE failed and rolled back. Amplify recreates this stack from"
      echo "  scratch on the next deploy, so there is usually NOTHING TO DELETE —"
      echo "  'NoStack: CloudFormationStack object does not hold a stack' in the"
      echo "  build log is the rollback that just happened, not an old stack in the"
      echo "  way. Fix the cause and redeploy."
      echo
      echo "  If the reason below is 'Validation failed with N error(s)' and there"
      echo "  is nothing more specific, that is template validation: CloudFormation"
      echo "  rejected the template before creating any resource, so there are no"
      echo "  per-resource events to read. DescribeEvents has nothing further —"
      echo "  the cause has to come from the template itself, not from here."
      ;;
    DELETE_FAILED|ROLLBACK_FAILED)
      echo
      echo "  ⚠  Stuck part-way through teardown. This one really does need"
      echo "     $0 delete-stack"
      ;;
    NOT_FOUND)
      echo "  Nothing to clean up. Next: $0 deploy"
      ;;
    *)
      echo "  Nothing obviously wrong with the stack itself."
      ;;
  esac

  # Why it failed. This is the only place the actual reason appears — the build
  # log just says "Validation failed with N error(s). Call DescribeEvents".
  #
  # A rolled-back stack is usually deleted straight after, and once it is,
  # describe-stack-events NO LONGER ACCEPTS THE NAME — only the full stack id.
  # Looking it up is the difference between reading the real cause and guessing
  # from the error count.
  echo
  echo "== failure events (the actual reason) =="
  local target="$STACK"
  if [ "$status" = "NOT_FOUND" ]; then
    target=$(aws cloudformation list-stacks \
               --stack-status-filter ROLLBACK_COMPLETE DELETE_COMPLETE CREATE_FAILED ROLLBACK_FAILED \
               --query "StackSummaries[?StackName=='${STACK}']|[0].StackId" --output text 2>/dev/null)
    if [ -z "$target" ] || [ "$target" = "None" ]; then
      echo "  no deleted stack found under that name either — nothing to read"
      return 0
    fi
    echo "  (stack is gone; reading events by id)"
  fi

  aws cloudformation describe-stack-events --stack-name "$target" \
      --query 'StackEvents[?ResourceStatusReason!=null && contains(ResourceStatus, `FAILED`)].[LogicalResourceId,ResourceStatusReason]' \
      --output text 2>/dev/null | head -20 \
    || echo "  (could not read events)"

  echo
  echo "== last Amplify job =="
  aws amplify list-jobs --app-id "$APP_ID" --branch-name "$BRANCH" --max-results 1 \
      --query 'jobSummaries[0].[jobId,status,commitId,endTime]' --output text 2>/dev/null \
    || echo "  (could not read jobs — check app id / branch)"
}

cmd_secrets() {
  require_aws
  # Amplify Gen2 keeps backend secrets in SSM Parameter Store. The exact path
  # differs between sandbox and branch deployments, so list rather than assume.
  #
  # The previous version of this piped into `|| echo "none found"`. With
  # `set -o pipefail` an AccessDenied from AWS also lands in that branch, so a
  # permissions problem printed as "no secrets exist" — and the line after it
  # then drew a conclusion from that. Two different states, one message, wrong
  # answer. Errors are now shown verbatim.
  echo "== SSM parameters under /amplify/ =="
  local out rc
  set +e
  out=$(aws ssm get-parameters-by-path --path "/amplify" --recursive \
          --query 'Parameters[].Name' --output text 2>&1)
  rc=$?
  set -e

  if [ $rc -ne 0 ]; then
    echo "  QUERY FAILED — this is NOT the same as 'no secrets':"
    echo "$out" | sed 's/^/    /'
    echo
    echo "  Add ssm:DescribeParameters + ssm:GetParametersByPath on"
    echo "  arn:aws:ssm:*:*:parameter/amplify/* (docs/ops/iam-vva-recover-readonly.json)"
  elif [ -z "$out" ] || [ "$out" = "None" ]; then
    echo "  (query succeeded and returned nothing — there really are no"
    echo "   parameters under /amplify/ in this account)"
  else
    echo "$out" | tr '\t' '\n' | sed 's/^/  /'
  fi

  # Probe the exact names directly. `--path` needs list permission; `get-parameter`
  # needs only read on that one name, so this often answers even when the listing
  # above is denied.
  echo
  echo "== direct probe of the paths this script writes =="
  for name in GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET; do
    local p="/amplify/${APP_ID}/${BRANCH}/${name}"
    set +e
    out=$(aws ssm get-parameter --name "$p" --query 'Parameter.Name' --output text 2>&1)
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
      echo "  EXISTS   $p"
    else
      echo "  MISSING  $p"
      echo "$out" | head -1 | sed 's/^/           /'
    fi
  done
}

cmd_set_secrets() {
  require_aws
  : "${GOOGLE_CLIENT_ID:?set GOOGLE_CLIENT_ID in the environment, not as an argument}"
  : "${GOOGLE_CLIENT_SECRET:?set GOOGLE_CLIENT_SECRET in the environment, not as an argument}"

  for name in GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET; do
    local path="/amplify/${APP_ID}/${BRANCH}/${name}"
    aws ssm put-parameter --name "$path" --type SecureString \
        --value "${!name}" --overwrite >/dev/null
    echo "  set $path"
  done
  echo "Verify with: $0 secrets"
}

cmd_delete_stack() {
  require_aws
  local status
  status=$(aws cloudformation describe-stacks --stack-name "$STACK" \
             --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND")
  if [ "$status" = "NOT_FOUND" ]; then
    echo "Stack already gone — nothing to do."
    return 0
  fi

  echo "About to DELETE CloudFormation stack:"
  echo "  $STACK   (status: $status)"
  echo
  echo "This destroys every resource in it — including the Cognito user pool for"
  echo "this branch, and every account inside it. It does NOT touch the sandbox."
  read -r -p "Type the branch name to confirm ($BRANCH): " confirm
  [ "$confirm" = "$BRANCH" ] || { echo "Aborted."; return 1; }

  aws cloudformation delete-stack --stack-name "$STACK"
  echo "Deleting; waiting for completion (this can take several minutes)…"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
  echo "Deleted. Next: make sure secrets and env vars are set, then $0 deploy"
}

cmd_set_env() {
  require_aws
  local origin="${WEB_APP_ORIGIN:-https://$(echo "$BRANCH" | tr '/_' '-').${APP_ID}.amplifyapp.com}"
  # Vite reads VITE_* at BUILD time and .env.local is gitignored, so a CI build
  # sees none of it unless it is set here.
  local api="${VITE_API_BASE_URL:-http://localhost:8000}"

  echo "WEB_APP_ORIGIN=$origin"
  echo "  (the BRANCH url — amplify/shared/origins.ts reads it at synth time to"
  echo "   build the CORS allow-list and the OAuth redirect URIs)"
  echo "VITE_API_BASE_URL=$api"
  echo "  (baked into the bundle at build time, not read at runtime)"

  # All variables go in ONE call: update-branch REPLACES the whole map, so
  # setting them one at a time silently deletes the others.
  aws amplify update-branch --app-id "$APP_ID" --branch-name "$BRANCH" \
      --environment-variables "WEB_APP_ORIGIN=$origin,VITE_API_BASE_URL=$api" >/dev/null
  echo "  applied to branch $BRANCH — takes effect on the NEXT build"
}

cmd_deploy() {
  require_aws

  # A push to the branch starts a build on its own, so `deploy` right after one
  # hits LimitExceededException: "already have pending or running jobs". That is
  # not a failure — the build being asked for is already underway. Attach to it
  # instead of erroring out, which is what anyone running this actually wants.
  local job running
  running=$(aws amplify list-jobs --app-id "$APP_ID" --branch-name "$BRANCH" --max-results 5 \
              --query 'jobSummaries[?status==`PENDING` || status==`RUNNING`]|[0].jobId' \
              --output text 2>/dev/null)

  if [ -n "$running" ] && [ "$running" != "None" ]; then
    job="$running"
    echo "job $job is already running (a push starts one automatically) — attaching"
  else
    job=$(aws amplify start-job --app-id "$APP_ID" --branch-name "$BRANCH" \
            --job-type RELEASE --query 'jobSummary.jobId' --output text)
    echo "started job $job"
  fi
  echo "polling every 20s"
  while true; do
    local st
    st=$(aws amplify get-job --app-id "$APP_ID" --branch-name "$BRANCH" --job-id "$job" \
           --query 'job.summary.status' --output text)
    echo "  $st"
    case "$st" in
      SUCCEED) echo "done: https://$(echo "$BRANCH" | tr '/_' '-').${APP_ID}.amplifyapp.com"; return 0 ;;
      FAILED|CANCELLED) echo "job $st — re-run '$0 status' for the failure events"; return 1 ;;
    esac
    sleep 20
  done
}

case "${1:-status}" in
  status)       cmd_status ;;
  secrets)      cmd_secrets ;;
  set-secrets)  cmd_set_secrets ;;
  delete-stack) cmd_delete_stack ;;
  set-env)      cmd_set_env ;;
  deploy)       cmd_deploy ;;
  *) sed -n '3,30p' "$0"; exit 2 ;;
esac
