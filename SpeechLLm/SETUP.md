# SpeechLLM Setup Guide

This guide provides step-by-step instructions for setting up the SpeechLLM environment on your machine.

## Prerequisites

- **Conda** or **Mamba** (recommended) installed
  - Download from: https://www.anaconda.com/download
  - Or use Mamba for faster installation: https://github.com/conda-forge/miniforge

## Quick Setup

### Option 1: Using Conda (Recommended)

```bash
# Navigate to the SpeechLLM directory
cd SpeechLLm

# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate speechllm

# Verify installation
python -c "import torch; import faster_whisper; print('✓ All packages imported successfully')"
```

### Option 2: Using Mamba (Faster)

```bash
cd SpeechLLm
mamba env create -f environment.yml
conda activate speechllm
```

### Option 3: Using requirements.txt with pip

If you prefer a plain pip installation (make sure Python 3.10.19 is installed):

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install from requirements.txt
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Troubleshooting

### Issue: `pyaudio` Installation Fails on Windows

**Cause**: PyAudio requires the native PortAudio library.

**Solution**:
```bash
# The environment.yml includes conda-forge portaudio, which should help.
# If pip still fails, try installing via pipwin on Windows:
pip install pipwin
pipwin install pyaudio
```

### Issue: `TTS` Import Errors or Version Conflicts

**Cause**: The package name can be ambiguous (case-sensitive on some systems).

**Solution**:
```bash
# Verify the correct package is installed
python -c "import TTS; print(TTS.__version__)"

# If you get an error, reinstall explicitly:
pip uninstall -y TTS tts
pip install TTS==0.22.0
```

### Issue: `soundfile` Import Fails

**Cause**: Missing native `libsndfile` C library.

**Solution**:
```bash
# Install the native library via conda
conda install -c conda-forge libsndfile pysoundfile

# Then reinstall Python soundfile package
pip install --force-reinstall soundfile==0.13.1
```

### Issue: `ollama` or `faster_whisper` Not Found

**Cause**: Package not installed or environment not activated.

**Solution**:
```bash
# Ensure the environment is activated:
conda activate speechllm

# Reinstall the package:
pip install --force-reinstall faster-whisper==1.2.1
pip install --force-reinstall ollama==0.6.1

# Verify:
python -c "import faster_whisper; import ollama; print('✓ OK')"
```

### Issue: GPU PyTorch Not Working

**Cause**: The environment.yml installs CPU-only PyTorch by default.

**Solution** (for NVIDIA GPUs):
```bash
# After creating the environment, install GPU-enabled PyTorch
# Visit https://pytorch.org/get-started/locally/ and select your configuration
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

Run this script to verify all core packages are working:

```bash
python << 'EOF'
print("Checking imports...")
try:
    import torch
    print(f"✓ torch {torch.__version__}")
    import faster_whisper
    print(f"✓ faster_whisper {faster_whisper.__version__}")
    import TTS
    print(f"✓ TTS {TTS.__version__}")
    import sounddevice
    print(f"✓ sounddevice {sounddevice.__version__}")
    import transformers
    print(f"✓ transformers {transformers.__version__}")
    print("\n✓ All imports successful!")
except Exception as e:
    print(f"✗ Error: {e}")
EOF
```

## Environment Contents

- **Python**: 3.10.19
- **Key Packages**:
  - `torch` 2.5.1 (deep learning framework)
  - `faster-whisper` 1.2.1 (speech-to-text)
  - `TTS` 0.22.0 (text-to-speech, Coqui)
  - `transformers` 4.38.2 (NLP models)
  - `librosa` 0.10.0 (audio analysis)
  - `sounddevice` 0.5.5 & `soundfile` 0.13.1 (audio I/O)
  - `ollama` 0.6.1 (local LLM client)
  - `tqdm` 4.67.3 (progress bars)

See [environment.yml](environment.yml) and [requirements.txt](requirements.txt) for the complete dependency list with pinned versions.

## Updating Dependencies

To update a single package:
```bash
pip install --upgrade <package_name>
```

To export your current environment (after making changes):
```bash
conda env export > environment.yml
```

## Support

If you encounter issues not covered here, check:
- The package's official documentation
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Coqui TTS Issues](https://github.com/coqui-ai/TTS)
- [Faster-Whisper Issues](https://github.com/SYSTRAN/faster-whisper)
