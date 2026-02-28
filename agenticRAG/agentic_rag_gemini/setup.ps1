# AgenticRAG System Setup Script for Windows PowerShell

Write-Host "🚀 Setting up AgenticRAG System Dependencies..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Check if in correct directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ Error: requirements.txt not found. Please run this from the project root." -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Blue
pip install -r requirements.txt

# Install system dependencies for OCR
Write-Host "🔧 Installing system dependencies for OCR..." -ForegroundColor Blue

# Check if Chocolatey is available for easy installation
if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host "Detected Chocolatey package manager..." -ForegroundColor Yellow
    choco install tesseract -y
    Write-Host "✅ Tesseract OCR installed via Chocolatey" -ForegroundColor Green
} else {
    Write-Host "⚠️  Chocolatey not found. Please install manually:" -ForegroundColor Yellow
    Write-Host "   1. Download Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor White
    Write-Host "   2. Add Tesseract to your PATH" -ForegroundColor White
    Write-Host "   3. Restart PowerShell" -ForegroundColor White
}

# Create necessary directories
Write-Host "📁 Creating data directories..." -ForegroundColor Blue
New-Item -ItemType Directory -Force -Path "data\vector_store" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Cyan
Write-Host "1. Copy .env.example to .env and add your API keys" -ForegroundColor White
Write-Host "2. Run: python run_ui.py" -ForegroundColor White
Write-Host "3. Open: http://localhost:8501" -ForegroundColor White
