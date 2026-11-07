"""Check statistical boundaries, exported artifacts, and upload failure modes."""

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from spambase.config import FEATURE_COLUMNS, METRIC_NAMES, MODEL_SPECS, ROOT, TARGET
from spambase.data import feature_groups
from spambase.evaluation import evaluate_model, read_test_csv, validate_test_frame


@pytest.fixture
def test_frame():
    return read_test_csv(ROOT / "test_data.csv")


def test_split_keeps_feature_groups_separate_and_meets_assignment_sizes(test_frame):
    train = read_test_csv(ROOT / "data/splits/train_data.csv")
    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    assert audit["raw_rows"] == 4601
    assert audit["exact_duplicates_removed"] == 391
    assert audit["conflicting_feature_groups_retained"] == 3
    assert len(train) + len(test_frame) == audit["clean_rows"] == 4210
    assert len(FEATURE_COLUMNS) == 57 >= 12
    assert len(train) == 3367 and len(test_frame) == 843
    combined = pd.concat([train, test_frame], ignore_index=True)
    assert not combined.duplicated().any()
    groups = feature_groups(combined)
    assert groups.nunique() == 4207
    assert not set(groups.iloc[:len(train)]) & set(groups.iloc[len(train):])
    assert combined.groupby(groups)[TARGET].nunique().gt(1).sum() == 3
    assert set(train[TARGET]) == set(test_frame[TARGET]) == {0, 1}
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    assert manifest["source_row_index"].is_unique
    assert manifest["partition"].value_counts().to_dict() == {"train": 3367, "test": 843}
    assert manifest.groupby("feature_group")["partition"].nunique().max() == 1


def test_identical_inputs_stay_in_one_model_selection_fold():
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv")
    folds = pd.read_csv(ROOT / "results/cv_fold_manifest.csv")
    assert set(folds["source_row_index"]) == set(manifest.loc[manifest["partition"] == "train", "source_row_index"])
    assert folds["source_row_index"].is_unique
    assert set(folds["validation_fold"]) == {0, 1, 2, 3, 4}
    assert folds.groupby("feature_group")["validation_fold"].nunique().max() == 1


def test_feature_order_matches_official_spambase_names():
    lines = (ROOT / "data/raw/spambase.names").read_text().splitlines()
    official_names = [line.split(":", 1)[0].strip() for line in lines
                      if ":" in line and not line.lstrip().startswith("|")]
    assert FEATURE_COLUMNS == official_names
    dictionary = pd.read_csv(ROOT / "data/feature_dictionary.csv")
    assert dictionary["name"].tolist() == FEATURE_COLUMNS + [TARGET]


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s["slug"])
def test_serialized_models_reproduce_exported_results_and_train_only_scaling(spec, test_frame):
    model = joblib.load(ROOT / "model" / f"{spec['slug']}.joblib")
    result = evaluate_model(model, test_frame)
    comparison = pd.read_csv(ROOT / "results/model_comparison.csv").set_index("ML Model Name")
    np.testing.assert_allclose(
        [result["metrics"][m] for m in METRIC_NAMES],
        comparison.loc[spec["name"], METRIC_NAMES].astype(float), atol=1e-9,
    )
    assert np.array(result["confusion_matrix"]).sum() == len(test_frame)
    assert len(result["predictions"]) == len(test_frame)
    assert np.isfinite(result["scores"]).all()
    assert all(0 <= p <= 1 for p in result["scores"])
    if spec["slug"] != "decision_tree":
        train = read_test_csv(ROOT / "data/splits/train_data.csv")
        np.testing.assert_allclose(model.named_steps["scale"].mean_, train[FEATURE_COLUMNS].mean(), atol=1e-10)
        assert model.named_steps["scale"].n_samples_seen_ == len(train)


def test_auc_uses_positive_class_probabilities_rather_than_predictions(test_frame):
    frame = pd.concat([test_frame[test_frame[TARGET] == label].head(2) for label in [0, 1]])

    class RankingModel:
        classes_ = np.array([1, 0])

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

        def predict_proba(self, features):
            positive = np.array([0.1, 0.2, 0.8, 0.9])
            return np.column_stack([positive, 1 - positive])

    metrics = evaluate_model(RankingModel(), frame)["metrics"]
    assert metrics["AUC"] == 1.0
    assert metrics["Accuracy"] == 0.5
    assert metrics["Precision"] == metrics["Recall"] == metrics["F1"] == 0


@pytest.mark.parametrize("label", [0, 1])
def test_one_class_auc_is_explicitly_undefined(test_frame, label):
    model = joblib.load(ROOT / "model/logistic_regression.joblib")
    result = evaluate_model(model, test_frame[test_frame[TARGET] == label])
    assert result["metrics"]["AUC"] is None
    assert np.asarray(result["confusion_matrix"]).shape == (2, 2)


def test_csv_accepts_reordered_columns_and_utf8_bom(test_frame):
    reordered = test_frame.loc[:, list(reversed(test_frame.columns))]
    uploaded = io.BytesIO(b"\xef\xbb\xbf" + reordered.to_csv(index=False).encode())
    pd.testing.assert_frame_equal(read_test_csv(uploaded), test_frame)


@pytest.mark.parametrize("mutation, message", [
    (lambda f: f.drop(columns=TARGET), "Missing columns"),
    (lambda f: f.assign(row_id=1), "Unexpected columns"),
    (lambda f: f.assign(spam=0.5), r"0 \(non-spam\) or 1 \(spam\)"),
    (lambda f: f.assign(spam="yes"), "must be numeric"),
    (lambda f: f.assign(word_freq_make="bad"), "must be numeric"),
    (lambda f: f.assign(word_freq_make=np.inf), "infinity"),
    (lambda f: f.assign(word_freq_make=np.nan), "Missing values"),
    (lambda f: f.iloc[:0], "at least one data row"),
])
def test_invalid_test_data_rejected(test_frame, mutation, message):
    with pytest.raises(ValueError, match=message):
        validate_test_frame(mutation(test_frame))


def test_csv_duplicate_headers_and_ragged_rows_rejected(test_frame):
    raw = test_frame.head(2).to_csv(index=False)
    header, rest = raw.split("\n", 1)
    duplicate = header.replace("word_freq_address", "word_freq_make") + "\n" + rest
    with pytest.raises(ValueError, match="Duplicate column"):
        read_test_csv(io.BytesIO(duplicate.encode()))
    ragged = header + "\n" + "0," + rest
    with pytest.raises(ValueError, match="same number of fields"):
        read_test_csv(io.BytesIO(ragged.encode()))


@pytest.mark.parametrize("raw", [b"", b"\xff\xfe", b"x;y\n1;2\n"])
def test_invalid_csv_encoding_or_structure_is_actionable(raw):
    with pytest.raises(ValueError):
        read_test_csv(io.BytesIO(raw))
