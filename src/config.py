"""Central configuration: paths and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sms_spam.tsv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "spam_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
