import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER = PROJECT_ROOT / "BitNet_Engine" / "train_engine_char_transformer.py"

runpy.run_path(str(TRAINER), run_name="__main__")
