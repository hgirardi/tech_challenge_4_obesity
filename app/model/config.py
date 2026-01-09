# config.py
from pathlib import Path

# Base do projeto: .../XX_Tech_Challenge
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "Obesity.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "model.joblib"

RANDOM_STATE = 150
TEST_SIZE = 0.20
THRESHOLD = 0.46