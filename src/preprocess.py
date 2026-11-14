"""Data loading and text preprocessing."""
import re
import string

import pandas as pd

# Compact English stop-word list (keeps the project free of NLTK downloads).
STOP_WORDS = frozenset("""
a about above after again all am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its itself just me
more most my myself no nor not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there these they this
those through to too under until up very was we were what when where which while who
whom why will with you your yours yourself u ur im ive id
""".split())

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")
NUM_RE = re.compile(r"\d+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lower-case, replace URLs/emails/phones/numbers with tokens, strip punctuation
    and stop-words. Replacing with tokens (rather than deleting) preserves strong
    spam signals such as 'contains a URL' or 'contains a long number'."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = text.translate(PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def load_dataset(path) -> pd.DataFrame:
    """Load the TSV (label<TAB>message), validate, drop duplicates, encode labels."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                     encoding="utf-8")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["target"] = (df["label"] == "spam").astype(int)
    df["clean"] = df["message"].apply(clean_text)
    return df.reset_index(drop=True)
