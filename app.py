"""Evaluate saved email spam classifiers in a local Streamlit application.

Run with ``streamlit run app.py`` after ``python train.py``. Training lives in
the separate training command; this app only loads pipelines and predicts.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spambase.config import FEATURE_COLUMNS, MODEL_SPECS, ROOT, TARGET
from spambase.evaluation import evaluate_model, read_test_csv


METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
CLASS_NAMES = ("Non-spam (0)", "Spam (1)")


@st.cache_resource(show_spinner=False)
def load_saved_models(signature: tuple[tuple[str, int], ...]) -> dict:
    """Reuse local saved pipelines, refreshing when an artifact is replaced.

    The signature is part of Streamlit's cache key. Only trusted artifacts from
    this project's model directory are loaded; CSV uploads are never unpickled.
    """
    return {
        spec["slug"]: joblib.load(ROOT / "model" / f"{spec['slug']}.joblib")
        for spec in MODEL_SPECS
    }


def model_signature() -> tuple[tuple[str, int], ...]:
    paths = [ROOT / "model" / f"{spec['slug']}.joblib" for spec in MODEL_SPECS]
    paths.append(ROOT / "model" / "metadata.json")
    return tuple((path.name, path.stat().st_mtime_ns) for path in paths)


def confusion_figure(matrix: list[list[int]]) -> plt.Figure:
    counts = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    axis.imshow(counts, cmap="Greens", vmin=0)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    for actual in range(2):
        for predicted in range(2):
            value = int(counts[actual, predicted])
            axis.text(
                predicted,
                actual,
                str(value),
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color="white" if value > counts.max() / 2 else "#153c2d",
            )
    axis.spines[:].set_visible(False)
    return figure


def report_frame(report: dict) -> pd.DataFrame:
    # sklearn's scalar accuracy entry is already displayed as a metric. Keep
    # class/average dictionaries so accuracy is not repeated as fake support.
    rows = {name: values for name, values in report.items() if isinstance(values, dict)}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame.rename(index={"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]})


def main() -> None:
    st.set_page_config(
        page_title="Spambase Model Lab",
        page_icon="📧",
        layout="wide",
    )
    st.html(
        "<style>[data-testid=\"stAppDeployButton\"] {display: none !important;}</style>"
    )

    st.title("Spambase Model Lab")
    st.markdown(
        "Compare five classifiers that detect **spam email** using word, "
        "character and capitalization statistics."
    )
    st.caption("UCI Spambase · Machine Learning Assignment 2")

    with st.sidebar:
        st.header("Run an evaluation")
        selected_name = st.selectbox(
            "Choose a classifier",
            options=[spec["name"] for spec in MODEL_SPECS],
            key="selected_model",
        )
        uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
        st.caption(
            f"Include all {len(FEATURE_COLUMNS)} input features and the `{TARGET}` "
            "target column. Use 0 for non-spam and 1 for spam."
        )
        example = ROOT / "test_data.csv"
        if example.is_file():
            st.download_button(
                "Download test CSV",
                data=example.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.divider()
        st.markdown("**Positive class: Spam (1)**")
        st.caption(
            "Each row describes an email using the dataset's numeric features. "
            "Each classifier uses its saved training pipeline; uploaded rows "
            "are used only for evaluation."
        )

    try:
        models = load_saved_models(model_signature())
    except (OSError, ValueError, ImportError) as error:
        st.error("The saved classifiers could not be loaded. Run `python train.py` first.")
        with st.expander("Load details"):
            st.code(str(error))
        st.stop()

    try:
        frame = read_test_csv(uploaded if uploaded is not None else example)
    except (OSError, ValueError, UnicodeError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        st.stop()

    source_name = uploaded.name if uploaded is not None else "Packaged held-out test data"
    st.info(
        f"{source_name} · {len(frame):,} rows · {len(FEATURE_COLUMNS)} input features · "
        f"{int((frame[TARGET] == 1).sum()):,} spam / "
        f"{int((frame[TARGET] == 0).sum()):,} non-spam"
    )
    try:
        with st.spinner("Evaluating saved classifiers…"):
            evaluations = {
                spec["slug"]: evaluate_model(models[spec["slug"]], frame)
                for spec in MODEL_SPECS
            }
    except (ValueError, TypeError) as error:
        st.error(f"The saved classifiers could not evaluate these rows: {error}")
        st.stop()

    selected = next(spec for spec in MODEL_SPECS if spec["name"] == selected_name)
    result = evaluations[selected["slug"]]
    st.subheader(selected_name)
    for column, metric in zip(st.columns(6), METRIC_NAMES, strict=True):
        value = result["metrics"][metric]
        column.metric(metric, "N/A" if value is None else f"{value:.4f}")

    if result["metrics"]["AUC"] is None:
        st.warning(
            "AUC is unavailable because this CSV contains only one target class. "
            "Upload test rows from both classes to calculate AUC."
        )

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        figure = confusion_figure(result["confusion_matrix"])
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with right:
        st.markdown("**Classification report**")
        st.dataframe(
            report_frame(result["classification_report"]),
            column_config={
                "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1-score": st.column_config.NumberColumn("F1", format="%.4f"),
                "support": st.column_config.NumberColumn("Support", format="%d"),
            },
            use_container_width=True,
        )
        predictions = frame.copy()
        predictions["predicted_spam"] = result["predictions"]
        predictions["probability_spam"] = result["scores"]
        st.download_button(
            "Download predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected['slug']}_predictions.csv",
            mime="text/csv",
        )

    st.subheader("All five models on the same test data")
    comparison = pd.DataFrame(
        [
            {"ML Model Name": spec["name"], **evaluations[spec["slug"]]["metrics"]}
            for spec in MODEL_SPECS
        ]
    )
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            metric: st.column_config.NumberColumn(metric, format="%.4f")
            for metric in METRIC_NAMES
        },
        use_container_width=True,
    )
    st.caption(
        "Every row uses the currently selected test CSV. Precision, recall and F1 "
        "refer to spam (1). AUC uses continuous model scores. MCC ranges from −1 to 1."
    )

    with st.expander("Understand the evaluation"):
        st.markdown(
            "- **Accuracy:** fraction of all predictions that are correct.\n"
            "- **AUC:** ability to rank spam above non-spam over different thresholds.\n"
            "- **Precision:** among predictions of spam, the fraction actually labelled spam.\n"
            "- **Recall:** fraction of actual spam emails detected.\n"
            "- **F1:** harmonic mean of spam precision and recall.\n"
            "- **MCC:** correlation between actual and predicted classes, including "
            "all four confusion matrix counts.\n\n"
            "The target is required for evaluation. The app loads saved models and "
            "their preprocessing steps; it does not train on this CSV."
        )
        st.caption(
            "Dataset: Hopkins, Reeber, Forman and Suermondt (1999), "
            "[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase). "
            "DOI: 10.24432/C53G6X. Licensed under CC BY 4.0."
        )


if __name__ == "__main__":
    main()
