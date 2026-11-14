"""Unit + integration tests.  Run: pytest -v"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.preprocess import clean_text, load_dataset
from src.predict import predict_message


# ---------- preprocessing ----------
def test_clean_lowercases_and_strips_punctuation():
    assert clean_text("HELLO, World!!!") == "hello world"

def test_clean_replaces_url_number_phone():
    out = clean_text("Visit http://spam.com now, call 09061701461 or text 2 win 100")
    assert "urltoken" in out and "phonetoken" in out and "numtoken" in out

def test_clean_removes_stopwords():
    assert "the" not in clean_text("the cat is on the mat").split()

def test_clean_handles_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == "none"  # str() coercion, must not crash

def test_load_dataset_schema_and_dedup():
    df = load_dataset(config.DATA_PATH)
    assert {"label", "message", "target", "clean"} <= set(df.columns)
    assert set(df.target.unique()) == {0, 1}
    assert not df.duplicated(subset=["label", "message"]).any()


# ---------- model & prediction ----------
@pytest.fixture(scope="module")
def metrics():
    return json.loads(config.METRICS_PATH.read_text())

def test_model_file_exists():
    assert config.MODEL_PATH.exists(), "Run `python -m src.train` first"

def test_model_quality_thresholds(metrics):
    best = metrics["test_results"][metrics["best_model"]]
    assert best["accuracy"] >= 0.96
    assert best["f1"] >= 0.88
    assert best["roc_auc"] >= 0.97

def test_obvious_spam_detected():
    r = predict_message("WINNER!! You have won a £1000 prize. Call 09061701461 now to claim your reward")
    assert r["label"] == "spam" and r["spam_probability"] > 0.5

def test_obvious_ham_detected():
    r = predict_message("Hey, are we still meeting for lunch tomorrow?")
    assert r["label"] == "ham" and r["spam_probability"] < 0.5

def test_output_schema_and_ranges():
    r = predict_message("ok see you soon")
    assert set(r) == {"label", "spam_probability", "confidence"}
    assert 0 <= r["spam_probability"] <= 1 and 0.5 <= r["confidence"] <= 1

@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        predict_message(bad)

def test_threshold_changes_decision():
    msg = "Free entry in a weekly competition, text WIN to 80086"
    assert predict_message(msg, threshold=0.0)["label"] == "spam"
    assert predict_message(msg, threshold=1.01)["label"] == "ham"


# ---------- web app ----------
def test_flask_endpoint():
    from app import app
    c = app.test_client()
    assert c.get("/").status_code == 200
    ok = c.post("/api/predict", json={"message": "Free prize! Call 09061701461 now"})
    assert ok.status_code == 200 and ok.get_json()["label"] in ("spam", "ham")
    assert c.post("/api/predict", json={"message": ""}).status_code == 400
