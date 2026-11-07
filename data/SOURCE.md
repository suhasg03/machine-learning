# Dataset provenance

This project uses **Spambase** from the UCI Machine Learning Repository.

- Dataset page: [UCI dataset 94](https://archive.ics.uci.edu/dataset/94/spambase).
- Official download: [Spambase ZIP](https://archive.ics.uci.edu/static/public/94/spambase.zip).
- Official metadata: [UCI dataset API, id 94](https://archive.ics.uci.edu/api/dataset?id=94).
- Citation: Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C53G6X](https://doi.org/10.24432/C53G6X).
- Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The source contains 4,601 email feature vectors: 2,788 non-spam and 1,813 spam. Inputs are 48 word-frequency percentages, six character-frequency percentages, and three capital-letter-run statistics. The final column is the target: `0 = non-spam`, `1 = spam`.

## Files retained

| File | Purpose |
| --- | --- |
| `raw/spambase.data` | Unchanged official headerless, comma-separated data; 57 inputs plus one target |
| `raw/spambase.names` | Original ordered feature definitions |
| `raw/spambase.DOCUMENTATION` | Original authors' description, counts, collection context and caveats |
| `raw/uci_metadata.json` | Retained official API response |
| `feature_dictionary.csv` | Original feature names with descriptions and target definition |
| `spambase.csv` | Named prepared dataset after exact-row deduplication |
| `splits/train_data.csv` | The 3,367 training rows |
| `../test_data.csv` | The 843 held-out evaluation rows |

SHA-256 of the retained original `raw/spambase.data`:

```text
b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12
```

## Missing-value discrepancy

The current UCI landing page flags missing values, while the bundled original `spambase.DOCUMENTATION` states `Missing Attribute Values: None`. Inspection of the retained 4,601-by-58 source file finds no missing or nonfinite values. This implementation uses that observed file audit, fits no imputer, and rejects missing/nonfinite test uploads. The original files and metadata are retained so this difference can be inspected.

## Adaptations and split policy

1. Assign the 57 original feature names from `spambase.names`, preserving their order, and name the final target column `spam`.
2. Validate finite numeric values and the existing 0/1 target encoding. The target is excluded from model inputs.
3. Remove 391 identical **full rows**, retaining the first original occurrence. This leaves 4,210 rows: 2,531 non-spam and 1,679 spam.
4. Preserve three input vectors with conflicting labels. Each has one remaining row of each class, so all six rows remain; labels are neither arbitrarily changed nor discarded.
5. Assign identical input vectors a shared group independently of their labels. There are 4,207 unique input groups. Take the first fold of five-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` as the held-out test portion. This gives 3,367 training rows and 843 test rows, approximately 80/20.
6. Keep those groups together in five-fold training-only model-selection CV as well. Fit preprocessing inside each pipeline and save the selected fitted pipelines.

Training contains 2,023 non-spam and 1,344 spam emails. Test data contains 508 non-spam and 335 spam emails. Grouping prevents identical input vectors from crossing train/test or training/validation boundaries even when their labels conflict.

`results/data_audit.json` records counts, removed source positions, duplicate policy and raw checksum. `results/split_manifest.csv` records zero-based source positions, feature groups and train/test membership. `results/cv_fold_manifest.csv` records the training groups' validation-fold assignments. These audit fields are excluded from features.

## Scope of the data

The dataset was collected in 1999. Its non-spam examples include personal/work email with collection-specific indicators such as `george`, `hp`, and `650`. A grouped split controls identical feature vectors but does not establish separation by sender, campaign or time, and this experiment does not establish performance on modern or unrelated email collections.

The app evaluates supplied feature CSVs; it does not parse raw email text or connect to mail accounts. Derived CSVs and the feature dictionary are attributed adaptations of the original dataset under CC BY 4.0.

`python train.py` reuses the bundled files offline after dependencies are installed. `python train.py --download` refreshes the official ZIP before preparing the experiment.
