#!/usr/bin/env python
"""
AgenticRAG UI Launcher - Start the web interface
Run this to launch the Streamlit UI: python run_ui.py
Or use: streamlit run ui.py
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"! Invalid integer for {name}={value!r}; using default {default}")
        return default

def main():
    """Launch the Streamlit UI."""
    parser = argparse.ArgumentParser(description="Launch AgenticRAG Streamlit UI")
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("AGENTICRAG_UI_PORT", 8501),
        help="UI port (default: env AGENTICRAG_UI_PORT or 8501)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("AGENTICRAG_UI_HOST", "0.0.0.0"),
        help="UI host/address (default: env AGENTICRAG_UI_HOST or 0.0.0.0)",
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 AgenticRAG - Web Interface Launcher")
    print("="*70)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✓ Streamlit found")
    except ImportError:
        print("✗ Streamlit not installed")
        print("\nInstall with: pip install streamlit streamlit-chat")
        sys.exit(1)
    
    # Get the UI file path
    ui_file = Path(__file__).parent / "ui.py"
    
    if not ui_file.exists():
        print(f"✗ UI file not found: {ui_file}")
        sys.exit(1)
    
    print(f"✓ UI file found: {ui_file}")
    
    print("\n📝 Configuration:")
    print(f"   Working directory: {Path(__file__).parent}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   URL: http://localhost:{args.port}")
    
    print("\n" + "-"*70)
    print("🌐 Launching web interface...")
    print("-"*70 + "\n")
    
    # Launch streamlit
    try:
        subprocess.run(
            [
                "streamlit",
                "run",
                str(ui_file),
                "--server.port",
                str(args.port),
                "--server.address",
                args.host,
                "--logger.level=info",
            ],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Web interface stopped")
    except Exception as e:
        print(f"\n✗ Error launching UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
