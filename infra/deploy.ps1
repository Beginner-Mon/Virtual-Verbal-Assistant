<#
.SYNOPSIS
    Build the CRUD package, then deploy the Track 2 stacks in an order that works.

.DESCRIPTION
    Four things this script exists to remember, all of which fail confusingly when
    done by hand.

    1. EVERY synth NEEDS infra/build/crud_api.zip.
       VvaCrudApiStack stopped being conditional on 20-08 — VvaRestApiStack
       references its function — so `cdk synth` of ANY stack now raises
       FileNotFoundError when the package is missing. Loud on purpose: the
       alternative is deploying an empty function.

    2. EVERY synth NEEDS -c agent_image_tag, even when VvaAgentStack is not being
       deployed. app.py constructs it unconditionally and it raises without one,
       so a command that omits the flag fails before it reaches the stack you
       asked for. Until 03-09 this script omitted it and could not deploy
       anything at all.

       The tag is discovered from the function that is actually running, so the
       agent stack synthesises to what is already deployed and is a no-op if it
       is ever included. Do NOT substitute a placeholder: agent_stack.py warns
       that a deploy which gets this wrong "would DELETE the live one, and
       CloudFormation would call that a success".

    3. --exclusively, on every stack.
       Without it, CDK pulls in dependency stacks — and VvaAssetStack refuses to
       synthesise without -c motion_public_key_file, whose .pub file is not in
       the repository. So a plain `cdk diff VvaRestApiStack` fails on a stack you
       were not asking about. Deploying exclusively is also the guarantee that a
       run touches only what is listed below.

    4. ORDER MATTERS while the Function URL is being removed.
       VvaAssetStack imported VvaCharacterStack's Function URL ARN, and
       CloudFormation refuses to delete an export that is still imported.
       Deploying VvaCharacterStack first failed with

           Cannot delete export ... as it is in use by VvaAssetStack

       and rolled back. That migration now appears to be finished — on 03-09
       VvaCharacterStack deployed on its own with no export change in the diff —
       but the order costs nothing and a rollback costs a lot.

    NEVER `cdk deploy --all`. Two stacks outside this list (VvaVpcStack,
    VvaKimodoEcsStack) are not part of a normal release; as of 03-09 both read
    UPDATE_COMPLETE, but --all would still attempt them for no reason.

.PARAMETER SkipBuild
    Reuse the existing infra/build/crud_api.zip instead of rebuilding it.

.PARAMETER Yes
    Skip cdk's interactive approval for IAM and security-group changes.
    The default is `--require-approval broadening`, which is right for a human at
    a terminal: it prints every widening IAM statement and waits. It also aborts
    with exit 1 in any non-interactive shell, which is what this switch is for.
    Review the changes with -WhatIf first; this only suppresses the prompt, not
    the diff.

.PARAMETER MotionPublicKeyFile
    Path to motion_signing_key.pub. Only VvaAssetStack needs it, and the file is
    deliberately not in the repository — so without this parameter that stack is
    skipped rather than failing the run. Everything else deploys without it.

.PARAMETER AgentImageTag
    Override the discovered tag. Only needed when the vva-agent function does not
    exist yet, or when deploying a new image on purpose.

.PARAMETER WhatIf
    Show `cdk diff` for each stack instead of deploying.

.EXAMPLE
    ./deploy.ps1 -WhatIf
    ./deploy.ps1
    ./deploy.ps1 -MotionPublicKeyFile motion_signing_key.pub
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$SkipBuild,
    [switch]$Yes,
    [string]$MotionPublicKeyFile,
    [string]$AgentImageTag
)

# Deliberately NOT 'Stop'. Windows PowerShell 5.1 turns any stderr line from a
# native command into a terminating NativeCommandError when it is, and `cdk`
# writes deprecation warnings to stderr on every run — so 'Stop' aborts the
# script on output that is not an error at all. Exit codes are checked explicitly
# after each call instead, which is the thing that actually indicates failure.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

# Ordered, and the order is the point. See note 4 above.
$stacks = @(
    'VvaAssetStack'      # dropped the Function URL import FIRST; needs the .pub
    'VvaCharacterStack'  # then the export could go
    'VvaCrudApiStack'
    'VvaRestApiStack'
)

# ── Context every synth needs ────────────────────────────────────────────────

function Get-LiveAgentImageTag {
    <#
      The tag on the image the deployed function is running. Reading it back
      rather than asking the caller to remember it is the whole point: the flag
      is mandatory for synth, and the only value that is safe by default is the
      one already in production.
    #>
    $uri = aws lambda get-function --function-name vva-agent --region us-east-1 `
        --query 'Code.ImageUri' --output text
    if ($LASTEXITCODE -ne 0) { return $null }
    if ([string]::IsNullOrWhiteSpace($uri)) { return $null }
    if ($uri -eq 'None') { return $null }
    # 244203483654.dkr.ecr.us-east-1.amazonaws.com/vva-agent:<sha> — the registry
    # host carries no colon, so the last segment is the tag.
    return ($uri.Trim() -split ':')[-1]
}

function Get-MotionKeyPairId {
    <#
      Assigned by CloudFront when VvaAssetStack first published the signing key.
      VvaAgentStack needs it to sign motion URLs; without it every
      GET /motion/{job_id} hands back a URL CloudFront answers with 403.
    #>
    $id = aws cloudfront list-public-keys `
        --query "PublicKeyList.Items[?contains(Name,'MotionSigningKey')].Id | [0]" `
        --output text
    if ($LASTEXITCODE -ne 0) { return $null }
    if ([string]::IsNullOrWhiteSpace($id)) { return $null }
    if ($id -eq 'None') { return $null }
    return $id.Trim()
}

if (-not $AgentImageTag) {
    Write-Host "-> reading the live agent image tag" -ForegroundColor Cyan
    $AgentImageTag = Get-LiveAgentImageTag
}
if (-not $AgentImageTag) {
    throw @'
Could not read the agent image tag, and every synth needs one.

Check that AWS credentials are configured (aws sts get-caller-identity), or pass
the tag yourself:

    ./deploy.ps1 -AgentImageTag <git-sha>

Pass the tag the live function is ALREADY running unless you mean to deploy a new
image. See agent_stack.py: a deploy that gets this wrong deletes the live
function and CloudFormation reports success.
'@
}
Write-Host "   agent_image_tag = $AgentImageTag" -ForegroundColor DarkGray

$motionKeyPairId = Get-MotionKeyPairId
if ($motionKeyPairId) {
    Write-Host "   motion_key_pair_id = $motionKeyPairId" -ForegroundColor DarkGray
} else {
    Write-Warning "motion_key_pair_id not found; VvaAgentStack would sign motion URLs CloudFront rejects. Harmless for the stacks below, which do not use it."
}

$context = @('-c', "agent_image_tag=$AgentImageTag")
if ($motionKeyPairId) { $context += @('-c', "motion_key_pair_id=$motionKeyPairId") }
if ($MotionPublicKeyFile) {
    if (-not (Test-Path $MotionPublicKeyFile)) {
        throw "-MotionPublicKeyFile '$MotionPublicKeyFile' does not exist"
    }
    $context += @('-c', "motion_public_key_file=$MotionPublicKeyFile")
}

# ── Build ────────────────────────────────────────────────────────────────────

if (-not $SkipBuild) {
    Write-Host "-> building infra/build/crud_api.zip" -ForegroundColor Cyan
    python (Join-Path $PSScriptRoot 'build_crud_api.py')
    if ($LASTEXITCODE -ne 0) { throw "build_crud_api.py failed" }
} elseif (-not (Test-Path (Join-Path $PSScriptRoot 'build/crud_api.zip'))) {
    throw "-SkipBuild was passed but infra/build/crud_api.zip does not exist. Run without it."
}

# ── Deploy ───────────────────────────────────────────────────────────────────

foreach ($stack in $stacks) {
    if ($stack -eq 'VvaAssetStack' -and -not $MotionPublicKeyFile) {
        Write-Host "`n-> skipping VvaAssetStack" -ForegroundColor Yellow
        Write-Host "   it cannot synthesise without -MotionPublicKeyFile, and that .pub is not in the repo." -ForegroundColor DarkGray
        Write-Host "   Pass it only when this stack actually needs deploying." -ForegroundColor DarkGray
        continue
    }

    if ($PSCmdlet.ShouldProcess($stack, 'cdk deploy')) {
        Write-Host "`n-> deploying $stack" -ForegroundColor Cyan
        $approval = if ($Yes) { 'never' } else { 'broadening' }
        npx cdk deploy $stack --exclusively --require-approval $approval @context
        if ($LASTEXITCODE -ne 0) {
            throw "$stack failed (exit $LASTEXITCODE); later stacks not attempted"
        }
    } else {
        Write-Host "`n-> diff $stack" -ForegroundColor Cyan
        npx cdk diff $stack --exclusively @context
    }
}

Write-Host "`nDone. Read the RestApiUrl output and set it as VITE_API_GATEWAY_URL" -ForegroundColor Green
Write-Host "on the Amplify 'release' branch, then redeploy that branch." -ForegroundColor Green
Write-Host "A new API Gateway route answers 'Missing Authentication Token' for a" -ForegroundColor DarkGray
Write-Host "minute or two after the deploy, exactly as an absent one does. Re-check" -ForegroundColor DarkGray
Write-Host "before concluding the deploy went wrong." -ForegroundColor DarkGray
