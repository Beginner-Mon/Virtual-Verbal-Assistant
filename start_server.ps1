Write-Host "Starting Virtual Verbal Assistant servers..."

$root = $PSScriptRoot

# AgenticRAG api server
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
cd '$root\agenticRAG\agentic_rag_gemini';
conda activate firstconda;
python api_server.py
"

# AgenticRAG main API
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
cd '$root\agenticRAG\agentic_rag_gemini';
conda activate firstconda;
python main_api.py
"

# Speech LLM
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
cd '$root\speechLLM';
conda activate tts;
python api_server.py
"

$dartWinPath = "$PSScriptRoot\text-to-motion\DART"
$dartWslPath = wsl wslpath -a "$dartWinPath"

Start-Process powershell -ArgumentList "-NoExit","-Command","wsl bash -ic `"cd '$dartWslPath'; pwd; exec bash`""