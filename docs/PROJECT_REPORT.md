# Spam Detection System – Project Report

## 1. Objective
Build an end-to-end spam classifier for text messages, with documented workflow, evaluation, testing and a usable prediction interface.

## 2. Data
- **Source:** SMS Spam Collection (5,574 labelled messages).
- **After cleaning:** 5,169 unique messages – 653 spam (12.6 %) and 4,516 ham (87.4 %).
- **Class imbalance:** spam is a minority class (see `reports/class_distribution.png`), so accuracy alone is misleading. A model that always says "ham" would score 87 %. We therefore track **precision, recall, F1 and ROC-AUC**, and use `class_weight="balanced"` where supported.
- 405 exact duplicates were removed so that identical messages cannot appear in both train and test sets.

## 3. Data preprocessing (`src/preprocess.py`)
1. Drop missing values and duplicates; normalise labels; encode `spam=1`, `ham=0`.
2. Lower-case the text.
3. Replace URLs, e-mails, long digit strings (phone/short codes) and other numbers with the tokens `urltoken`, `emailtoken`, `phonetoken`, `numtoken`. Replacing instead of deleting keeps strong spam signals ("contains a phone number").
4. Remove punctuation and English stop-words; drop 1-character tokens.
5. **Feature extraction:** TF-IDF with unigrams + bigrams, `min_df=2`, sublinear term frequency. Vectorisation lives *inside* the scikit-learn `Pipeline`, so it is fitted only on training folds (no leakage).

## 4. Model training (`src/train.py`)
- Stratified 80/20 train/test split (`random_state=42`).
- Four candidates compared with stratified 5-fold CV (F1 on spam) on the training set only: Multinomial Naive Bayes, Logistic Regression, Linear SVM (probability-calibrated), Random Forest.
- Best CV model (Logistic Regression, F1 = 0.950 ± 0.006) is evaluated once on the test set and saved with `joblib`.

## 5. Evaluation
Test set: 1,034 messages (903 ham, 131 spam).

Confusion matrix (Logistic Regression):

|  | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 899 | 4 |
| **Actual spam** | 9 | 122 |

- Accuracy 98.74 %, spam precision 96.83 %, spam recall 93.13 %, F1 94.94 %, ROC-AUC 0.997.
- Only 4 legitimate messages were flagged as spam (false positives), which matters in practice because losing a real message is usually worse than seeing a spam one.
- Most informative spam features: `phonetoken`, `urltoken`, `call phonetoken`, `text`, `txt`, `reply`, `mobile`, `free`, `ringtone`, `claim`, `stop`.
- Plots: `reports/confusion_matrix.png`, `reports/roc_curves.png`. Misclassified examples are stored in `reports/metrics.json` for error analysis.

### Limitations
- Trained on SMS data from ~2005 (largely UK/Singapore); modern spam (phishing links, crypto, delivery scams) and other languages / e-mail may perform worse.
- Only 131 spam messages in the test set, so metrics have noticeable variance (a single message changes recall by ~0.8 pp).
- Bag-of-words ignores word order and context.

### Future work
Retrain on a larger, newer corpus; try transformer embeddings (e.g. DistilBERT); add hand-crafted features (message length, capital ratio, currency symbols); tune the decision threshold for the desired precision/recall trade-off; add model monitoring and user feedback loop.

## 6. Prediction interface
- **CLI:** `python -m src.predict "text"` or interactive mode.
- **Web UI:** `python app.py` – paste a message, get a verdict and probability bar.
- **JSON API:** `POST /api/predict`.
- The decision threshold (default 0.5) is adjustable in `predict_message(msg, threshold=...)`.

## 7. Testing (`tests/test_spam.py`, 16 tests – all passing; see `reports/test_results.txt`)
- Preprocessing: casing, punctuation, URL/number tokens, stop-words, empty/odd input, dataset schema and de-duplication.
- Model: file exists, minimum quality gates (accuracy ≥ 0.96, F1 ≥ 0.88, AUC ≥ 0.97), obvious spam/ham examples, output schema, threshold behaviour.
- Robustness: empty / whitespace / `None` / non-string input raises `ValueError`.
- Web app: page loads, valid prediction, 400 on invalid input.

## 8. Reproducing results
```bash
pip install -r requirements.txt
python -m src.train && python -m pytest -v
```
Random seeds are fixed in `src/config.py`.

## 9. Tools & skills demonstrated
Python, pandas, scikit-learn (Pipeline, TF-IDF, cross-validation, calibration), NLP text cleaning, imbalanced-data evaluation, matplotlib, Flask, pytest, joblib, project structuring and documentation.
