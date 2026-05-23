"""Phase 0 smoke test runner — adds langgraph_agents to path."""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_project_root = _here.parents[1]
_agentic_root = _project_root / "agenticRAG" / "agentic_rag_gemini"
sys.path.insert(0, str(_agentic_root))

# Import and run tests programmatically
import pytest

sys.exit(pytest.main([str(_here / "test_phase0_smoke.py"), "-v"]))
