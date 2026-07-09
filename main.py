import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from matchlens.pipeline import run_phase1_pipeline

if __name__ == "__main__":
    run_phase1_pipeline()
