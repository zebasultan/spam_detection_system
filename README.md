# Spam Detection System

A complete, working machine-learning project that classifies SMS messages as **spam** or **ham** (legitimate).
It covers the full ML workflow: data preprocessing → model training & selection → evaluation → prediction interface → testing.

## Results (held-out test set, 1,034 messages)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| Naive Bayes | 98.45% | 97.52% | 90.08% | 93.65% | 0.994 | 0.933 |
| **Logistic Regression (selected)** | **98.74%** | **96.83%** | **93.13%** | **94.94%** | **0.997** | **0.950** |
| Linear SVM | 98.74% | 97.58% | 92.37% | 94.90% | 0.997 | 0.948 |
| Random Forest | 99.13% | 99.19% | 93.89% | 96.47% | 0.996 | 0.946 |

The model is selected using **cross-validation on the training split only**; the test set is touched once, for the final report (this avoids test-set leakage).
Random Forest scores slightly higher on this particular test split, but the difference is within CV noise (±0.013), so the simpler, faster, more interpretable Logistic Regression was kept.

## Quick start

```bash
pip install -r requirements.txt

python -m src.train                 # preprocess, train, evaluate, save model + reports
python -m src.predict "WINNER! Claim your free prize now, call 09061701461"   # CLI
python -m src.predict               # interactive CLI
python app.py                       # web UI at http://127.0.0.1:5000
python -m pytest -v                 # run the test suite (16 tests)
```

A pre-trained model (`models/spam_model.joblib`) is included, so prediction works immediately.

## Project structure

```
spam_detection_system/
├── data/sms_spam.tsv        SMS Spam Collection (5,574 rows; 5,169 after de-duplication)
├── src/
│   ├── config.py            paths & hyper-parameters
│   ├── preprocess.py        loading, cleaning, label encoding
│   ├── train.py             CV model selection, evaluation, plots, model export
│   └── predict.py           prediction API + CLI
├── app.py                   Flask web app + JSON API
├── templates/index.html     web interface
├── tests/test_spam.py       pytest suite
├── models/spam_model.joblib trained pipeline (TF-IDF + classifier)
├── reports/                 metrics.json, confusion_matrix.png, roc_curves.png, class_distribution.png, test_results.txt
└── docs/                    PROJECT_REPORT.md (workflow documentation), SUBMISSION_CHECKLIST.md
```

## Web API

`POST /api/predict` with `{"message": "..."}` returns
`{"label": "spam", "spam_probability": 0.9985, "confidence": 0.9985}`.

## Dataset

SMS Spam Collection (Almeida & Hidalgo, UCI ML Repository), 747 spam / 4,827 ham messages originally. Downloaded from a public GitHub mirror (`justmarkham/pycon-2016-tutorial`). Please cite the original authors if you publish results.
