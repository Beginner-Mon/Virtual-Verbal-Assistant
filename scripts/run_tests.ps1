<#
.SYNOPSIS
    Run one module's test suite in the conda environment that owns its deps.

.DESCRIPTION
    Calls conda by absolute path rather than by name. `conda` is only on PATH in
    a shell that has run `conda init`, and ~/.condarc here sets
    auto_activate: false — so a bare `conda` works in an interactive terminal and
    fails in a script, in CI, or in an agent session. Resolving the path removes
    that difference.

.PARAMETER Module
    langgraph | speech | dart

.PARAMETER Unit
    Only run tests marked `unit` — no database, no network, no LLM calls.

.EXAMPLE
    .\scripts\run_tests.ps1 langgraph -Unit
    .\scripts\run_tests.ps1 langgraph tests/langgraph_agents/test_auth.py -v
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('langgraph', 'speech', 'dart')]
    [string]$Module,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs,

    [switch]$Unit
)

$ErrorActionPreference = 'Stop'

if (-not $Module) {
    Write-Host "Usage: .\scripts\run_tests.ps1 <langgraph|speech|dart> [pytest args] [-Unit]"
    Write-Host "Example: .\scripts\run_tests.ps1 langgraph -Unit"
    exit 1
}

function Resolve-Conda {
    $candidates = @(
        'C:\Miniconda\Scripts\conda.exe',
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        'C:\ProgramData\miniconda3\Scripts\conda.exe'
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }

    $onPath = Get-Command conda -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    throw "conda not found. Looked in: $($candidates -join ', ')"
}

# langgraph and speech run on Windows; dart needs WSL for its CUDA-linked deps.
$suites = @{
    'langgraph' = @{ Env = 'firstconda'; Path = 'tests/langgraph_agents/';   Wsl = $false }
    'speech'    = @{ Env = 'tts';        Path = 'tests/SpeechLLm/';          Wsl = $false }
    'dart'      = @{ Env = 'DART';       Path = 'tests/text-to-motion/DART/'; Wsl = $true }
}
$suite = $suites[$Module]

$pytestArgv = @()
if ($PytestArgs) { $pytestArgv += $PytestArgs } else { $pytestArgv += $suite.Path }
if ($Unit) { $pytestArgv += @('-m', 'unit') }

if ($suite.Wsl) {
    $wslRoot = (wsl wslpath -a (Get-Item '.').FullName).Trim()
    $cmd = "source ~/miniconda3/etc/profile.d/conda.sh; conda activate $($suite.Env); " +
           "cd '$wslRoot'; python -m pytest $($pytestArgv -join ' ')"
    Write-Host "[$Module] via WSL / conda $($suite.Env)"
    wsl -e bash -lc $cmd
    exit $LASTEXITCODE
}

$conda = Resolve-Conda
Write-Host "[$Module] $conda run -n $($suite.Env) -- python -m pytest $($pytestArgv -join ' ')"

# `python -m pytest`, not `pytest`: the module form uses the environment's own
# interpreter, so it cannot pick up a pytest shim from elsewhere on PATH.
& $conda run -n $suite.Env --no-capture-output python -m pytest @pytestArgv
exit $LASTEXITCODE
