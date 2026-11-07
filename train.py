"""Train all five classifiers and write reproducible local artifacts."""

import argparse

from spambase.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Refresh the bundled raw data from UCI first.")
    args = parser.parse_args()
    run_training(download=args.download)


if __name__ == "__main__":
    main()
