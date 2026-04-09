"""
evaluate.py — Evaluate trained models on the held-out spatial test set.

Outputs written to outputs/reports/
------------------------------------
classification_report_rf.txt
classification_report_svm.txt
confusion_matrix_rf.png
confusion_matrix_svm.png
roc_auc_rf.png
roc_auc_svm.png
feature_importance_rf.png       (RF only — SVM has no native feature importances)
metrics_summary.csv             (collated per-class precision / recall / F1 + AUC)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    RF_MODEL_PATH, SVM_MODEL_PATH,
    REPORT_DIR, DATA_PROCESSED,
    FEATURE_NAMES, CLASS_NAMES, RANDOM_STATE,
)
from src.modeling.sample import load_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CLASSES      = sorted(CLASS_NAMES.keys())          # [1, 2, 3]
CLASS_LABELS = [CLASS_NAMES[c] for c in CLASSES]  # ["Low", "Moderate", "High"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_test_set() -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = load_samples()
    test_mask = np.load(DATA_PROCESSED / "test_mask.npy")
    return X[test_mask], y[test_mask]


def _save_classification_report(y_true, y_pred, name: str) -> dict:
    report_str = classification_report(
        y_true, y_pred,
        labels       = CLASSES,
        target_names = CLASS_LABELS,
    )
    path = REPORT_DIR / f"classification_report_{name}.txt"
    path.write_text(report_str)
    log.info("[%s] Classification report → %s", name, path)
    log.info("\n%s", report_str)

    report_dict = classification_report(
        y_true, y_pred,
        labels       = CLASSES,
        target_names = CLASS_LABELS,
        output_dict  = True,
    )
    return report_dict


def _save_confusion_matrix(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name.upper()}")
    fig.tight_layout()
    path = REPORT_DIR / f"confusion_matrix_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("[%s] Confusion matrix → %s", name, path)


def _save_roc_curves(y_true, y_prob, name: str) -> float:
    """
    One-vs-rest ROC curves for each class.
    Returns macro-average AUC.
    """
    y_bin = label_binarize(y_true, classes=CLASSES)   # (n, 3)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#f4d03f", "#e67e22", "#c0392b"]
    aucs = []
    for i, (cls, label, color) in enumerate(zip(CLASSES, CLASS_LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
        aucs.append(auc_val)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc_val:.3f})")

    macro_auc = float(np.mean(aucs))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) — {name.upper()}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = REPORT_DIR / f"roc_auc_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("[%s] ROC curves (macro AUC = %.4f) → %s", name, macro_auc, path)
    return macro_auc


def _save_feature_importance(model, name: str) -> None:
    """Extract feature importances from the RF classifier inside the pipeline."""
    try:
        clf = model.named_steps["clf"]
        importances = clf.feature_importances_
    except AttributeError:
        log.warning("[%s] No feature_importances_ attribute — skipping.", name)
        return

    order = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in order]
    sorted_vals  = importances[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1],
                   color="#2980b9", edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Feature Importances — {name.upper()}")
    for bar, val in zip(bars, sorted_vals[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    path = REPORT_DIR / f"feature_importance_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("[%s] Feature importance chart → %s", name, path)


def _evaluate_one(
    name:       str,
    model_path: Path,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
) -> dict:
    log.info("=" * 60)
    log.info("Evaluating %s", name)

    model   = joblib.load(model_path)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)    # (n, 3)

    report   = _save_classification_report(y_test, y_pred, name)
    _save_confusion_matrix(y_test, y_pred, name)
    macro_auc = _save_roc_curves(y_test, y_prob, name)
    _save_feature_importance(model, name)

    # Collate summary row
    summary = {"model": name, "macro_auc": macro_auc}
    for label in CLASS_LABELS:
        for metric in ("precision", "recall", "f1-score"):
            key = f"{label}_{metric.replace('-', '_')}"
            summary[key] = report[label][metric]
    summary["accuracy"] = report["accuracy"]
    return summary


def run() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading test set …")
    X_test, y_test = _load_test_set()
    log.info("Test set shape: X=%s  y=%s", X_test.shape, y_test.shape)

    rows = []
    # for name, path in [("rf", RF_MODEL_PATH), ("svm", SVM_MODEL_PATH)]:
    for name, path in [("rf", RF_MODEL_PATH)]:
        summary = _evaluate_one(name, path, X_test, y_test)
        rows.append(summary)

    df = pd.DataFrame(rows).set_index("model")
    csv_path = REPORT_DIR / "metrics_summary.csv"
    df.to_csv(csv_path)
    log.info("Metrics summary → %s", csv_path)
    log.info("\n%s", df.to_string())

    log.info("Evaluation complete.")


if __name__ == "__main__":
    run()
