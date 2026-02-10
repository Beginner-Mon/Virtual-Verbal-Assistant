#!/bin/bash

echo "🚀 Setting up AgenticRAG System Dependencies..."
echo "=========================================="

# Check if in correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this from the project root."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install system dependencies for OCR
echo "🔧 Installing system dependencies for OCR..."

# Check if running on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "Detected Ubuntu/Debian system..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
    sudo apt-get install -y poppler-utils  # For PDF rendering
    echo "✅ OCR dependencies installed"
else
    echo "⚠️  Non-Ubuntu system detected. Please install manually:"
    echo "   - tesseract-ocr (OCR engine)"
    echo "   - poppler-utils (PDF rendering)"
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/vector_store
mkdir -p logs

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy .env.example to .env and add your API keys"
echo "2. Run: python run_ui.py"
echo "3. Open: http://localhost:8501"
