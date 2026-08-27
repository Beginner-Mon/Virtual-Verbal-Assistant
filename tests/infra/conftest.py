import sys
from pathlib import Path

# Make the `infra` package (infra/infra/*.py, alongside infra/app.py, infra/cdk.json)
# importable as `infra.kimodo_ecs_stack` etc. Root pytest.ini's static `pythonpath`
# list covers agenticRAG/SpeechLLm/DART/kimodo — CDK stacks live in a sibling tree
# with the same nested-package shape, so this suite gets its own conftest rather
# than growing the shared list for a directory only these tests import.
_project_root = Path(__file__).resolve().parents[2]
_infra_root = _project_root / "infra"
sys.path.insert(0, str(_infra_root))
