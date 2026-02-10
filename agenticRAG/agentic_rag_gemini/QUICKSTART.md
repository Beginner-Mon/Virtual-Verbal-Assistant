# Quick Start Guide - Gemini Version

## 5-Minute Setup

### Step 1: Get Gemini API Key (2 minutes)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Select project or create new one
4. Copy your API key (starts with `AIza...`)

### Step 2: Install (2 minutes)

```bash
cd agentic_rag_gemini
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure (1 minute)

```bash
cp .env.example .env
nano .env  # Add: GEMINI_API_KEY=your_key_here
```

### Step 4: Run!

```bash
mkdir -p data/vector_store logs
python main.py --mode interactive
```

## Example Usage

```
You: I have neck pain from sitting all day
Assistant: I understand. Let me suggest some neck stretches...

You: What did we discuss before?
Assistant: [Retrieves context from memory]
```

That's it! 

API KEY Update:
# 1. Go to https://aistudio.google.com/app/apikey
# 2. Create a new API key (different from previous)
# 3. Update your .env file:
# Windows (PowerShell):
notepad .env
# OR: code .env
 
# Linux/Mac:
nano .env
# Replace: GEMINI_API_KEY=your_new_key_here