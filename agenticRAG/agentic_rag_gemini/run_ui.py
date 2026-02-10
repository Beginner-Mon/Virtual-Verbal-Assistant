#!/usr/bin/env python
"""
AgenticRAG UI Launcher - Start the web interface
Run this to launch the Streamlit UI: python run_ui.py
Or use: streamlit run ui.py
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit UI."""
    
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
    print(f"   Port: 8501 (default)")
    print(f"   URL: http://localhost:8501")
    
    print("\n" + "-"*70)
    print("🌐 Launching web interface...")
    print("-"*70 + "\n")
    
    # Launch streamlit
    try:
        subprocess.run(
            ["streamlit", "run", str(ui_file), "--logger.level=info"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Web interface stopped")
    except Exception as e:
        print(f"\n✗ Error launching UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
