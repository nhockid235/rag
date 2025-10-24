#!/bin/bash

# Professional RAG System - Launch Script
# ========================================

echo "🚀 Starting Professional RAG System..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import sentence_transformers, faiss, gradio, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Dependencies not found. Installing..."
    pip install -r requirements.txt
fi

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
curl -s http://localhost:11434/api/tags > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️ Ollama not running. Please start Ollama first:"
    echo "   ollama serve"
    echo "   ollama pull gpt-oss:20b"
    echo ""
    echo "Continuing without Ollama (some features may not work)..."
fi

# Launch the system
echo "🌐 Launching Professional RAG System..."
echo "📍 URL: http://localhost:7860"
echo ""

python rag_product.py --host 0.0.0.0 --port 7860 --share
