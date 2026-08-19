<#
.SYNOPSIS
    Mint a Cognito ID token for a development user, for curl and Postman.

.DESCRIPTION
    The API takes identity from a verified Bearer token and from nowhere else —
    there is no flag that turns that off, in any environment (see
    agenticRAG/langgraph_agents/api/auth.py). Through the browser that is
    invisible: Amplify holds the token and the axios interceptor attaches it.
    Poking the API by hand is where you need one, and copying it out of DevTools
    every hour is not a workflow.

    This talks to the SANDBOX pool created by `npx ampx sandbox`, not production.
    That separation is the whole design: dev and production differ by which user
    pool they trust, so a token from here is rejected by production because the
    issuer does not match.

.PARAMETER PoolId
    Cognito user pool id. Defaults to $env:COGNITO_USER_POOL_ID.

.PARAMETER ClientId
    App client id. Defaults to $env:COGNITO_APP_CLIENT_ID.

.PARAMETER Username
    Dev user. Defaults to $env:VVA_DEV_USERNAME.

.PARAMETER Password
    Dev password. Defaults to $env:VVA_DEV_PASSWORD.

.EXAMPLE
    $env:TOKEN = & scripts\dev_token.ps1
    curl "http://localhost:8001/sessions" -H "Authorization: Bearer $env:TOKEN"

.NOTES
    One-time setup — ADMIN_USER_PASSWORD_AUTH is not enabled by default on an
    Amplify-generated app client:

        aws cognito-idp update-user-pool-client `
            --user-pool-id  <pool> --client-id <client> `
            --explicit-auth-flows ALLOW_ADMIN_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH

    And a user to log in as:

        aws cognito-idp admin-create-user --user-pool-id <pool> --username dev@local
        aws cognito-idp admin-set-user-password `
            --user-pool-id <pool> --username dev@local --password '<pw>' --permanent
#>

[CmdletBinding()]
param(
    [string]$PoolId   = $env:COGNITO_USER_POOL_ID,
    [string]$ClientId = $env:COGNITO_APP_CLIENT_ID,
    [string]$Username = $env:VVA_DEV_USERNAME,
    [string]$Password = $env:VVA_DEV_PASSWORD,
    [string]$Region   = $env:COGNITO_REGION
)

$ErrorActionPreference = 'Stop'

# Named individually rather than as one "something is missing": the caller has
# four separate things to go and find, and a single message means discovering
# them one failed run at a time.
$missing = @()
if (-not $PoolId)   { $missing += 'PoolId (COGNITO_USER_POOL_ID)' }
if (-not $ClientId) { $missing += 'ClientId (COGNITO_APP_CLIENT_ID)' }
if (-not $Username) { $missing += 'Username (VVA_DEV_USERNAME)' }
if (-not $Password) { $missing += 'Password (VVA_DEV_PASSWORD)' }
if ($missing.Count -gt 0) {
    throw "Missing: $($missing -join ', '). Pass them as parameters or set the env vars."
}

if ($PoolId -notmatch '^[a-z]{2}-[a-z]+-\d_') {
    throw "PoolId '$PoolId' does not look like a Cognito pool id (e.g. us-east-1_ab12CD34)."
}

$awsArgs = @(
    'cognito-idp', 'admin-initiate-auth',
    '--user-pool-id', $PoolId,
    '--client-id', $ClientId,
    '--auth-flow', 'ADMIN_USER_PASSWORD_AUTH',
    '--auth-parameters', "USERNAME=$Username,PASSWORD=$Password",
    '--query', 'AuthenticationResult.IdToken',
    '--output', 'text'
)
if ($Region) { $awsArgs += @('--region', $Region) }

$token = & aws @awsArgs

if ($LASTEXITCODE -ne 0 -or -not $token -or $token -eq 'None') {
    throw @"
Could not mint a token. The usual causes, in order of likelihood:
  * ADMIN_USER_PASSWORD_AUTH not enabled on the app client — see the NOTES in
    this script for the update-user-pool-client call that enables it.
  * The user is in FORCE_CHANGE_PASSWORD. Set a permanent password with
    admin-set-user-password --permanent.
  * Wrong pool/client pair: the client must belong to the pool.
"@
}

# Bare token on stdout so `$env:TOKEN = & scripts\dev_token.ps1` just works.
# Anything else — a banner, a "success" line — ends up inside the header.
$token.Trim()
