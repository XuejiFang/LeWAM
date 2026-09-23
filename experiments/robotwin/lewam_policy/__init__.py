from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .deploy_policy import eval, get_model, reset_model  # noqa: E402,F401

__all__ = ["eval", "get_model", "reset_model"]
