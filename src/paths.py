from pathlib import Path
import os

# directory paths
PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
MODELS_DIR = PARENT_DIR / 'models'
RESULTS_DIR = PARENT_DIR / 'results'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

if not Path(RESULTS_DIR).exists():
    os.mkdir(RESULTS_DIR)