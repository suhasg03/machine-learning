"""Reproducible train-only model selection and one shared hold-out evaluation."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from spambase.config import FEATURE_COLUMNS, MODEL_SPECS, ROOT, SEED, TARGET, TEST_SIZE
from spambase.data import feature_groups, prepare_data
from spambase.evaluation import evaluate_model


def model_candidates() -> dict:
    """Small explicit grids keep local training and later lab runs practical."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=SEED),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"classifier__max_depth": [4, 8, None], "classifier__min_samples_leaf": [2, 8]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [5, 11, 21], "classifier__weights": ["uniform", "distance"]},
        ),
        "naive_bayes": (
            GaussianNB(), {"classifier__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        ),
        "svm": (
            SVC(kernel="rbf", probability=True, random_state=SEED),
            {"classifier__C": [0.5, 2.0, 8.0], "classifier__gamma": ["scale", 0.01]},
        ),
    }


def write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_figures(test: pd.DataFrame, evaluations: dict) -> None:
    figures = ROOT / "results/figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(8.5, 6), layout="constrained")
    for spec in MODEL_SPECS:
        result = evaluations[spec["slug"]]
        fpr, tpr, _ = roc_curve(test[TARGET], result["scores"])
        axis.plot(fpr, tpr, linewidth=2,
                  label=f"{spec['name']} (AUC {result['metrics']['AUC']:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="#888888", label="Chance ranking")
    axis.set(xlabel="False positive rate (non-spam classified as spam)", ylabel="True positive rate (spam recall)",
             title="ROC curves on the shared held-out test set", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.18)
    figure.savefig(figures / "roc_curves.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for axis, spec in zip(axes.flat, MODEL_SPECS):
        result = evaluations[spec["slug"]]
        display = ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["Non-spam (0)", "Spam (1)"])
        display.plot(ax=axis, colorbar=False, cmap="Greens", values_format="d")
        axis.set_title(spec["name"])
    axes.flat[-1].set_axis_off()
    figure.suptitle(f"Confusion matrices: {len(test)} held-out samples")
    figure.savefig(figures / "confusion_matrices.png", dpi=170)
    plt.close(figure)


def run_training(download: bool = False) -> pd.DataFrame:
    for name in ["model", "results", "results/cv", "data/splits"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    clean, audit = prepare_data(download=download)
    groups = feature_groups(clean)
    # The first of five grouped folds is the fixed ~20% hold-out. Conflicting
    # labels stay in the data, but identical inputs cannot straddle partitions.
    outer = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=SEED)
    train_positions, test_positions = next(outer.split(clean[FEATURE_COLUMNS], clean[TARGET], groups))
    train, test = clean.iloc[train_positions].copy(), clean.iloc[test_positions].copy()
    train_groups = groups.iloc[train_positions]
    # Keep source row positions outside the feature CSVs: they are audit IDs, not inputs.
    manifest = pd.DataFrame({"source_row_index": clean.index, "feature_group": groups.to_numpy(),
                             "partition": ["train" if i in train.index else "test" for i in clean.index]})
    manifest.to_csv(ROOT / "results/split_manifest.csv", index=False)
    train.to_csv(ROOT / "data/splits/train_data.csv", index=False)
    test.to_csv(ROOT / "test_data.csv", index=False)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    validation_folds = np.empty(len(train), dtype=int)
    for fold, (_, validation_positions) in enumerate(cv.split(train[FEATURE_COLUMNS], train[TARGET], train_groups)):
        validation_folds[validation_positions] = fold
    pd.DataFrame({"source_row_index": train.index, "feature_group": train_groups.to_numpy(),
                  "validation_fold": validation_folds}).to_csv(ROOT / "results/cv_fold_manifest.csv", index=False)
    comparison, cv_summary, evaluations, model_details = [], [], {}, {}
    predictions = test[[TARGET]].reset_index(drop=True).copy()

    for spec in MODEL_SPECS:
        slug, name = spec["slug"], spec["name"]
        classifier, grid = model_candidates()[slug]
        scaler = "passthrough" if slug == "decision_tree" else StandardScaler()
        pipeline = Pipeline([("scale", scaler), ("classifier", clone(classifier))])
        print(f"Training {name}: 5-fold grouped CV on {len(train)} training rows…", flush=True)
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=1, refit=True,
                              error_score="raise", return_train_score=False)
        search.fit(train[FEATURE_COLUMNS], train[TARGET], groups=train_groups)
        fitted = search.best_estimator_
        joblib.dump(fitted, ROOT / "model" / f"{slug}.joblib", compress=3)
        result = evaluate_model(fitted, test)
        evaluations[slug] = result
        comparison.append({"ML Model Name": name, **result["metrics"]})
        best_std = float(search.cv_results_["std_test_score"][search.best_index_])
        cv_summary.append({"ML Model Name": name, "CV F1 Mean": float(search.best_score_), "CV F1 Std": best_std})
        table = pd.DataFrame(search.cv_results_)
        selected_columns = [c for c in table if c.startswith("param_")] + [
            "mean_test_score", "std_test_score", "rank_test_score"]
        table[selected_columns].to_csv(ROOT / "results/cv" / f"{slug}.csv", index=False)
        model_details[slug] = {"name": name, "best_parameters": search.best_params_,
                               "cv_f1_mean": float(search.best_score_), "cv_f1_std": best_std,
                               "scaled": slug != "decision_tree"}
        predictions[f"{slug}_prediction"] = result["predictions"]
        predictions[f"{slug}_probability_spam"] = result["scores"]
        print(f"  CV F1={search.best_score_:.4f}; test F1={result['metrics']['F1']:.4f}", flush=True)

    comparison = pd.DataFrame(comparison)
    cv_summary = pd.DataFrame(cv_summary)
    # This choice uses training folds only; do not tune again after seeing test metrics.
    selected_slug = max(model_details, key=lambda s: model_details[s]["cv_f1_mean"])
    selected_model = model_details[selected_slug]["name"]
    comparison.to_csv(ROOT / "results/model_comparison.csv", index=False, float_format="%.10f")
    cv_summary.to_csv(ROOT / "results/cross_validation_summary.csv", index=False, float_format="%.10f")
    predictions.to_csv(ROOT / "results/test_predictions.csv", index=False)
    write_json(ROOT / "results/evaluation_details.json", evaluations)
    write_json(ROOT / "results/data_audit.json", audit)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Spambase", "dataset_doi": "10.24432/C53G6X",
        "features": FEATURE_COLUMNS, "target": TARGET, "class_mapping": {"non-spam": 0, "spam": 1},
        "positive_class": 1, "random_state": SEED, "test_size_fraction": TEST_SIZE,
        "actual_test_size_fraction": len(test) / len(clean),
        "split_strategy": "First fold of seeded 5-fold StratifiedGroupKFold; identical feature vectors share a group.",
        "train_rows": len(train), "test_rows": len(test),
        "train_class_counts": {str(k): int(v) for k, v in train[TARGET].value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in test[TARGET].value_counts().items()},
        "selection": "highest mean spam-class F1 in stratified grouped 5-fold CV on training data",
        "selected_model_slug": selected_slug, "selected_model_name": selected_model,
        "metric_convention": "spam=1 positive; binary precision/recall/F1; AUC from P(spam)",
        "data_audit": audit, "models": model_details,
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
        "test_csv_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
    }
    write_json(ROOT / "model/metadata.json", metadata)
    save_figures(test, evaluations)
    print("\nShared held-out test results:\n" + comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nModel selected by training CV: {selected_model}")
    return comparison
