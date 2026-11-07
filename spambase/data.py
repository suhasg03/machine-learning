"""Load the original Spambase source and audit repeated email feature vectors."""

import hashlib
import io
import zipfile
from urllib.request import Request, urlopen

import pandas as pd

from spambase.config import DOWNLOAD_URL, FEATURE_COLUMNS, ROOT, TARGET
from spambase.evaluation import validate_test_frame


def feature_groups(frame: pd.DataFrame) -> pd.Series:
    """Give identical input vectors one group ID, independently of their labels."""
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(frame[FEATURE_COLUMNS]), sort=False)
    return pd.Series(groups, index=frame.index, name="feature_group")


def prepare_data(download: bool = False) -> tuple[pd.DataFrame, dict]:
    """Deduplicate complete rows; retain conflicting labels for grouped splitting."""
    raw_path = ROOT / "data/raw/spambase.data"
    if download or not raw_path.exists():
        request = Request(DOWNLOAD_URL, headers={"User-Agent": "ML-Assignment-2/1.0"})
        try:
            with urlopen(request, timeout=60) as response:
                archive = response.read()
        except OSError as exc:
            raise RuntimeError(
                "UCI download failed. Keep the bundled data/raw/spambase.data or "
                "download the official ZIP listed in data/SOURCE.md."
            ) from exc
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            for name in ["spambase.data", "spambase.names", "spambase.DOCUMENTATION"]:
                (raw_path.parent / name).write_bytes(zipped.read(name))
    frame = pd.read_csv(raw_path, header=None)
    if frame.shape != (4601, 58):
        raise ValueError(f"Unexpected UCI source shape {frame.shape}; expected (4601, 58).")
    frame.columns = FEATURE_COLUMNS + [TARGET]
    frame = validate_test_frame(frame)
    original_counts = {str(k): int(v) for k, v in frame[TARGET].value_counts().items()}
    duplicate_mask = frame.duplicated(keep="first")
    clean = frame.loc[~duplicate_mask].copy()
    groups = feature_groups(clean)
    conflicting = clean.groupby(groups)[TARGET].nunique().gt(1)
    clean.to_csv(ROOT / "data/spambase.csv", index=False)
    audit = {
        "raw_rows": len(frame), "feature_count": len(FEATURE_COLUMNS),
        "source_class_counts": original_counts, "missing_values": int(frame.isna().sum().sum()),
        "exact_duplicates_removed": int(duplicate_mask.sum()),
        "removed_source_row_indices_zero_based": frame.index[duplicate_mask].tolist(),
        "repeated_feature_rows_in_source": int(frame.duplicated(subset=FEATURE_COLUMNS).sum()),
        "conflicting_feature_groups_retained": int(conflicting.sum()),
        "unique_feature_groups": int(groups.nunique()),
        "duplicate_policy": "Remove identical full rows; keep conflicting labels and group identical inputs in all outer/model-selection splits.",
        "clean_rows": len(clean),
        "clean_class_counts": {str(k): int(v) for k, v in clean[TARGET].value_counts().items()},
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    return clean, audit
