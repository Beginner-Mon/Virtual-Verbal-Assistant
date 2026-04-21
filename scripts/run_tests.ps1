param(
    [string]$Module = ""
)

if ($Module -eq "") {
    Write-Host "Please specify a module to test: dart, speech, or agenticrag"
    Write-Host "Example: .\scripts\run_tests.ps1 dart"
    exit
}

$Module = $Module.ToLower()
$RootPath = (Get-Item -Path ".\").FullName

if ($Module -eq "dart") {
    Write-Host "Running DART tests in WSL via conda..."
    $WslRoot = (wsl wslpath -a $RootPath).Trim()
    $Cmd = "source ~/miniconda3/etc/profile.d/conda.sh; conda activate DART; cd '$WslRoot'; pytest tests/text-to-motion/DART/ -v"
    wsl -e bash -lc $Cmd
}
elseif ($Module -eq "speech") {
    Write-Host "Running SpeechLLM tests via conda on Windows..."
    conda run -n tts pytest tests/SpeechLLm/ -v
}
elseif ($Module -eq "agenticrag") {
    Write-Host "Running AgenticRAG tests via conda on Windows..."
    conda run -n firstconda pytest tests/agenticRAG/ -v
}
else {
    Write-Host "Unknown module: $Module"
}
