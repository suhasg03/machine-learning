"""The app and training script use exactly the same validation and metrics."""

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from spambase.config import FEATURE_COLUMNS, TARGET

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEST_ROWS = 20_000


def validate_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Require all 57 finite numeric features and a binary 0/1 spam target."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The test CSV must contain a header and at least one data row.")
    if len(frame) > MAX_TEST_ROWS:
        raise ValueError(f"Upload at most {MAX_TEST_ROWS:,} test rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    required = FEATURE_COLUMNS + [TARGET]
    missing = [c for c in required if c not in frame.columns]
    extra = [str(c) for c in frame.columns if c not in required]
    if missing or extra:
        issues = []
        if missing:
            issues.append("Missing columns: " + ", ".join(missing))
        if extra:
            issues.append("Unexpected columns: " + ", ".join(extra))
        raise ValueError("; ".join(issues) + ". Download the sample CSV for the required schema.")
    try:
        ordered = frame.loc[:, required].apply(pd.to_numeric, errors="raise")
        values = ordered.to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("All features must be numeric; spam must be 0 or 1.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Missing values, NaN and infinity are not allowed in the test CSV.")
    if not ordered[TARGET].isin([0, 1]).all():
        raise ValueError("spam must use 0 (non-spam) or 1 (spam).")
    ordered[TARGET] = ordered[TARGET].astype(int)
    return ordered


def read_test_csv(source) -> pd.DataFrame:
    """Read a path or uploaded file, checking headers before pandas can rename them."""
    if isinstance(source, (str, Path)):
        if Path(source).stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("The test CSV must be at most 10 MB.")
        raw = Path(source).read_bytes()
    else:
        source.seek(0)
        raw = source.read(MAX_UPLOAD_BYTES + 1)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The test CSV must be at most 10 MB.")
    try:
        text = raw.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not allowed.")
        # Reject ragged rows rather than silently interpreting a feature as an index.
        for row in csv.reader(io.StringIO(text)):
            if row and len(row) != len(header):
                raise ValueError("Every CSV row must have the same number of fields as its header.")
        frame = pd.read_csv(io.StringIO(text))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise ValueError("Upload a valid UTF-8, comma-separated CSV with a header row.") from exc
    return validate_test_frame(frame)


def evaluate_model(model, frame: pd.DataFrame) -> dict:
    """Evaluate a fitted pipeline; never call fit or fit_transform here."""
    valid = validate_test_frame(frame)
    features, target = valid[FEATURE_COLUMNS], valid[TARGET]
    predictions = model.predict(features)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(features)[:, positive_index]
    auc = float(roc_auc_score(target, scores)) if target.nunique() == 2 else None
    # sklearn defines MCC as 0 when one input is constant. Handle this explicitly
    # so a valid one-class upload does not emit sklearn's confusion-matrix warning.
    mcc = (
        float(matthews_corrcoef(target, predictions))
        if target.nunique() == 2 and np.unique(predictions).size == 2 else 0.0
    )
    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)), "AUC": auc,
        "Precision": float(precision_score(target, predictions, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(target, predictions, pos_label=1, zero_division=0)),
        "F1": float(f1_score(target, predictions, pos_label=1, zero_division=0)),
        "MCC": mcc,
    }
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            target, predictions, labels=[0, 1], target_names=["Non-spam (0)", "Spam (1)"],
            output_dict=True, zero_division=0,
        ),
        "predictions": predictions.astype(int).tolist(), "scores": scores.astype(float).tolist(),
    }
