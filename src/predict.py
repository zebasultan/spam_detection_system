"""Prediction API + command-line interface."""
import argparse
import sys
from functools import lru_cache

import joblib

from . import config
from .preprocess import clean_text


@lru_cache(maxsize=1)
def _load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run: python -m src.train")
    return joblib.load(config.MODEL_PATH)


def predict_message(message: str, threshold: float = 0.5) -> dict:
    """Classify one message. Returns label, spam probability and confidence."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")
    model = _load_model()
    p_spam = float(model.predict_proba([clean_text(message)])[0, 1])
    is_spam = p_spam >= threshold
    return {"label": "spam" if is_spam else "ham",
            "spam_probability": round(p_spam, 4),
            "confidence": round(p_spam if is_spam else 1 - p_spam, 4)}


def predict_batch(messages, threshold: float = 0.5):
    return [predict_message(m, threshold) for m in messages]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Spam detector CLI")
    ap.add_argument("message", nargs="*", help="Message text (omit for interactive mode)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    if args.message:
        r = predict_message(" ".join(args.message), args.threshold)
        print(f"{r['label'].upper()}  (spam probability {r['spam_probability']:.2%})")
        return 0
    print("Spam detector – type a message (empty line or Ctrl+C to quit)")
    try:
        while True:
            text = input("> ").strip()
            if not text:
                break
            r = predict_message(text, args.threshold)
            print(f"  -> {r['label'].upper()}  (spam probability {r['spam_probability']:.2%})")
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
