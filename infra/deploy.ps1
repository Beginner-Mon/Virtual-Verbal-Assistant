<#
.SYNOPSIS
    Build the CRUD package, then deploy the Track 2 stacks in an order that works.

.DESCRIPTION
    Two things this script exists to remember, both of which fail confusingly when
    done by hand.

    1. EVERY synth NEEDS infra/build/crud_api.zip.
       VvaCrudApiStack stopped being conditional on 20-08 — VvaRestApiStack
       references its function — so `cdk synth` of ANY stack now raises
       FileNotFoundError when the package is missing. Loud on purpose: the
       alternative is deploying an empty function.

    2. ORDER MATTERS while the Function URL is being removed.
       VvaAssetStack currently imports VvaCharacterStack's Function URL ARN, and
       CloudFormation refuses to delete an export that is still imported. Deploying
       VvaCharacterStack first fails with

           Cannot delete export ... as it is in use by VvaAssetStack

       and rolls back. AssetStack has to drop the import before CharacterStack can
       drop the export. After that migration this ordering no longer matters —
       there is no cross-stack reference left — but the script keeps it because a
       rollback costs more than a fixed order does.

    NEVER `cdk deploy --all`: VvaVpcStack and VvaKimodoEcsStack have sat in
    UPDATE_ROLLBACK_COMPLETE since 28-07 (an ECS service was deleted outside
    CloudFormation) and --all would retry them.

.PARAMETER SkipBuild
    Reuse the existing infra/build/crud_api.zip instead of rebuilding it.

.PARAMETER WhatIf
    Show `cdk diff` for each stack instead of deploying.

.EXAMPLE
    ./deploy.ps1 -WhatIf
    ./deploy.ps1
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $PSScriptRoot

# Ordered, and the order is the point. See the note above.
$stacks = @(
    'VvaAssetStack'      # drops the Function URL import FIRST
    'VvaCharacterStack'  # then the export can go
    'VvaCrudApiStack'
    'VvaRestApiStack'
)

if (-not $SkipBuild) {
    Write-Host "→ building infra/build/crud_api.zip" -ForegroundColor Cyan
    python (Join-Path $PSScriptRoot 'build_crud_api.py')
    if ($LASTEXITCODE -ne 0) { throw "build_crud_api.py failed" }
} elseif (-not (Test-Path (Join-Path $PSScriptRoot 'build/crud_api.zip'))) {
    throw "-SkipBuild was passed but infra/build/crud_api.zip does not exist. Run without it."
}

foreach ($stack in $stacks) {
    if ($PSCmdlet.ShouldProcess($stack, 'cdk deploy')) {
        Write-Host "`n→ deploying $stack" -ForegroundColor Cyan
        npx cdk deploy $stack --require-approval broadening
        if ($LASTEXITCODE -ne 0) { throw "$stack failed; later stacks not attempted" }
    } else {
        Write-Host "`n→ diff $stack" -ForegroundColor Cyan
        npx cdk diff $stack
    }
}

Write-Host "`nDone. Read the RestApiUrl output and set it as VITE_API_GATEWAY_URL" -ForegroundColor Green
Write-Host "on the Amplify 'release' branch, then redeploy that branch." -ForegroundColor Green
