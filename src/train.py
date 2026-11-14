"""Train several models, compare them with cross-validation, evaluate the best on a
held-out test set, and save the model + metrics + plots."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from . import config
from .preprocess import load_dataset


def build_candidates():
    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(
            C=10, max_iter=1000, class_weight="balanced"))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=3))]),
        "Random Forest": Pipeline([("tfidf", tfidf()), ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1,
            random_state=config.RANDOM_STATE))]),
    }


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    df = load_dataset(config.DATA_PATH)
    print(f"Loaded {len(df)} unique messages | spam ratio = {df.target.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=config.TEST_SIZE,
        stratify=df["target"], random_state=config.RANDOM_STATE)
    raw_test = df.loc[X_test.index, "message"]

    # ---- Model selection with stratified 5-fold CV on the training split only ----
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = {}
    for name, pipe in build_candidates().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_results[name] = {"mean_f1": float(scores.mean()), "std_f1": float(scores.std())}
        print(f"  CV F1  {name:<20} {scores.mean():.4f} +/- {scores.std():.4f}")
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_f1"])
    print(f"Best model by CV: {best_name}")

    # ---- Final fit + held-out evaluation ----
    candidates = build_candidates()
    test_results = {}
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        test_results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        }
    best = fitted[best_name]
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    print("\nHeld-out test report for", best_name)
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["ham", "spam"]
                           ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix – {best_name}")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, pipe in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={test_results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    counts = df["label"].value_counts()
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
    ax.set(title="Class distribution", ylabel="Messages")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # ---- Top spam-indicating words (from the linear model) ----
    top_words = []
    lr = fitted["Logistic Regression"]
    names = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    coefs = lr.named_steps["clf"].coef_[0]
    top_words = names[np.argsort(coefs)[-15:][::-1]].tolist()

    # ---- Misclassified examples for error analysis ----
    wrong = raw_test[pred != y_test]
    errors = [{"message": m, "true": "spam" if t else "ham", "predicted": "spam" if p else "ham"}
              for m, t, p in zip(wrong, y_test[pred != y_test], pred[pred != y_test])]

    metrics = {
        "dataset": {"messages": int(len(df)), "spam": int(df.target.sum()),
                    "ham": int((1 - df.target).sum())},
        "best_model": best_name,
        "cross_validation": cv_results,
        "test_results": test_results,
        "best_confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "top_spam_words": top_words,
        "misclassified_examples": errors[:20],
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Save the deployable model (trained on the train split; same object evaluated above)
    joblib.dump(best, config.MODEL_PATH)
    print(f"\nSaved model -> {config.MODEL_PATH}\nSaved metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
