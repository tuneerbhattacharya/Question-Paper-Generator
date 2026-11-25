import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import langgraph.graph  # noqa: F401  (real library, e.g. in CI)
except ImportError:
    from tests import fake_langgraph
    pkg = types.ModuleType("langgraph")
    pkg.graph = fake_langgraph
    sys.modules["langgraph"] = pkg
    sys.modules["langgraph.graph"] = fake_langgraph
