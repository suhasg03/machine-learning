"""Exercise the app with real local artifacts and Streamlit's AppTest runner."""

from __future__ import annotations

import joblib
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

import spambase.evaluation as evaluation
from spambase.config import MODEL_SPECS, ROOT, TARGET


def run_app() -> AppTest:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    return app


def test_default_app_displays_actual_metrics_and_all_five_models():
    app = run_app()
    assert app.title[0].value == "Spambase Model Lab"
    assert not app.error
    assert {metric.label for metric in app.metric} == {
        "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
    }
    comparison = next(
        table.value for table in app.dataframe if "ML Model Name" in table.value.columns
    )
    assert comparison["ML Model Name"].tolist() == [spec["name"] for spec in MODEL_SPECS]
    frame = evaluation.read_test_csv(ROOT / "test_data.csv")
    model = joblib.load(ROOT / "model" / f"{MODEL_SPECS[0]['slug']}.joblib")
    expected = evaluation.evaluate_model(model, frame)["metrics"]
    for metric in app.metric:
        assert metric.value == f"{expected[metric.label]:.4f}"


def test_switching_models_loads_saved_pipelines_without_training(monkeypatch):
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("The evaluation app must not fit a model.")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    app = run_app()
    frame = evaluation.read_test_csv(ROOT / "test_data.csv")
    for spec in MODEL_SPECS:
        app.selectbox[0].select(spec["name"]).run()
        assert not app.exception
        assert not app.error
        assert app.subheader[0].value == spec["name"]
        model = joblib.load(ROOT / "model" / f"{spec['slug']}.joblib")
        expected = evaluation.evaluate_model(model, frame)["metrics"]
        for metric in app.metric:
            assert metric.value == f"{expected[metric.label]:.4f}"


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_test_data_shows_auc_as_unavailable(monkeypatch, label):
    frame = evaluation.read_test_csv(ROOT / "test_data.csv")
    one_class = frame.loc[frame[TARGET] == label].head(12).copy()
    monkeypatch.setattr(evaluation, "read_test_csv", lambda source: one_class)
    app = run_app()
    assert not app.error
    assert next(metric for metric in app.metric if metric.label == "AUC").value == "N/A"
    assert any("one target class" in warning.value for warning in app.warning)
    comparison = next(
        table.value for table in app.dataframe if "ML Model Name" in table.value.columns
    )
    assert comparison["AUC"].isna().all()


def test_invalid_csv_displays_validation_error(monkeypatch):
    def reject_csv(source):
        raise ValueError("Missing required columns: spam")

    monkeypatch.setattr(evaluation, "read_test_csv", reject_csv)
    app = run_app()
    assert any("Missing required columns: spam" in error.value for error in app.error)
    assert len(app.metric) == 0
