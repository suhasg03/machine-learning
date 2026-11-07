"""Evaluate the saved models on a labelled CSV without training."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from spambase.config import MODEL_SPECS, ROOT
from spambase.evaluation import evaluate_model, read_test_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "test_data.csv")
    parser.add_argument("--model", choices=[s["slug"] for s in MODEL_SPECS], help="Default: evaluate all five.")
    parser.add_argument("--output", type=Path, help="Optional location for a comparison CSV.")
    args = parser.parse_args()
    try:
        frame = read_test_csv(args.csv)
        rows = []
        for spec in MODEL_SPECS:
            if args.model and args.model != spec["slug"]:
                continue
            fitted = joblib.load(ROOT / "model" / f"{spec['slug']}.joblib")
            rows.append({"ML Model Name": spec["name"], **evaluate_model(fitted, frame)["metrics"]})
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="N/A"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
