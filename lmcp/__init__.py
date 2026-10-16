"""lmcp: a small harness for critical-period experiments on tiny language models (MLX, Apple silicon)."""
from pathlib import Path

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
QUEUE = ROOT / "queue"
RESULTS = ROOT / "results"
