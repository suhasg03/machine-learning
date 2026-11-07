# Understanding the Spambase experiment

Use this guide alongside the Python source to review and reproduce the work. The assignment allows built-in scikit-learn functions. Its AI guidance permits learning support, so review the code and be able to explain the dataset, split, metrics and observations before submitting.

## 1. Define the prediction task

Each row represents an email through 57 numeric statistics: word-frequency percentages, character-frequency percentages, and capital-letter-run lengths. The target `spam` is 1 for spam and 0 for non-spam. This is binary supervised classification.

The project uses the supplied statistics. It does not extract them from raw email text, connect to email accounts or implement a live mailbox filter. The app accepts the labelled feature CSV so predictions can be evaluated against known targets.

`spambase/config.py` provides the shared feature names, target, model labels, random seed and file paths. The target, original row positions and feature-group IDs are excluded from predictors.

## 2. Audit duplicates and preserve ambiguous examples

`spambase/data.py` reads `data/raw/spambase.data`, assigns the official feature names and validates values. There are no missing or nonfinite values in the retained raw file. The current UCI metadata flag differs from the original documentation; `data/SOURCE.md` records that discrepancy and the file audit.

The source has 391 exact repeated full rows, which are removed before splitting. There are also three distinct input vectors with conflicting target labels. Removing repeated full rows leaves one row of each label for each such vector. All six ambiguous rows remain, because their target labels cannot be resolved from the supplied inputs.

`feature_groups()` gives identical input vectors the same group ID without using target labels. The resulting 4,210 rows have 4,207 unique input groups. Grouping is required because ordinary row-level splitting could put the same inputs on both sides of an evaluation boundary.

## 3. Create a fixed grouped hold-out and training folds

`spambase/training.py` uses the first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` as the shared held-out set. Stratification tries to preserve the class proportions while respecting the groups. This produces 3,367 training rows and 843 test rows, approximately 80/20 rather than an exact row-count split.

Training has 2,023 non-spam and 1,344 spam rows; testing has 508 non-spam and 335 spam rows. All five classifiers use this same test set. The outer hold-out is not used to choose hyperparameters.

A separate five-fold `StratifiedGroupKFold` provides training-only validation folds to `GridSearchCV`. Identical inputs remain in one fold. Original row positions and group IDs are recorded in the split and CV-fold manifests for audit purposes and are never supplied as model features.

## 4. Fit preprocessing and select models

Features have different numeric scales. Scaling helps Logistic Regression optimisation, KNN distances, and the SVM RBF kernel. `StandardScaler` uses training means and standard deviations to transform inputs.

Scaling is inside the saved `Pipeline` for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree uses original values because monotonic standard scaling does not change split ordering. Each CV fit learns scaling only from its training fold. After selecting parameters, `GridSearchCV(refit=True)` fits the chosen pipeline on the full 3,367-row training portion.

| Classifier           | Learning approach                                                         | Parameters explored                                          |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Logistic Regression  | Linear decision boundary in scaled feature space                          | Regularisation strength `C`; optional balanced class weights |
| Decision Tree        | Sequence of learned feature thresholds                                    | Maximum depth; minimum samples per leaf                      |
| KNN                  | Labels of nearby training samples                                         | Neighbour count; uniform or distance weighting               |
| Gaussian Naive Bayes | Per-feature Gaussian likelihoods with conditional-independence assumption | Variance smoothing                                           |
| SVM                  | Nonlinear margin-based boundary with an RBF kernel                        | `C`; kernel-width parameter `gamma`                          |

The search score is F1 for spam, the minority positive class. Candidate grids are explicit in the source, and their scores are saved in `results/cv/`. The overall selected model is the one with the highest mean training CV F1. Fold standard deviation describes variation for the selected configuration, not a confidence interval or proof of statistical superiority.

SVM enables probability estimation for AUC and export. Predicted labels come from `predict`; calibrated SVC probabilities need not give identical labels when thresholded at exactly 0.5. The grouped folds described above are the outer split and model-selection folds; SVC performs its probability calibration internally.

## 5. Interpret the six metrics

Spam is positive. A true positive (TP) is spam correctly detected. A false positive (FP) is non-spam incorrectly flagged as spam. A true negative (TN) is non-spam correctly recognised, and a false negative (FN) is spam that passes through as non-spam.

| Metric    | Meaning                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------ |
| Accuracy  | `(TP + TN) / total`: fraction of correct predictions                                                   |
| Precision | `TP / (TP + FP)`: fraction of spam predictions that are actually spam                                  |
| Recall    | `TP / (TP + FN)`: fraction of actual spam detected                                                     |
| F1        | Harmonic mean of spam precision and recall                                                             |
| ROC AUC   | Ability of continuous spam scores to rank spam above non-spam across thresholds                        |
| MCC       | Correlation of actual and predicted labels using all four confusion-matrix counts; ranges from -1 to 1 |

An always-non-spam prediction obtains 0.6026 accuracy on this test split while detecting no spam. The other metrics expose that limitation. False positives are also significant: they represent legitimate email being flagged as spam, so compare precision and the actual FP counts alongside F1.

The code uses zero for undefined precision/recall/F1 divisions and returns MCC as zero when either label array is constant, following scikit-learn's convention. AUC is undefined if the uploaded actual targets contain only one class; the app shows `N/A` with an explanation.

Confusion-matrix rows are actual labels and columns are predicted labels, ordered non-spam (0), spam (1): `[[TN, FP], [FN, TP]]`. The classification report includes class-specific and averaged metrics. Required comparison-table precision, recall and F1 use the spam class specifically, not macro or weighted averaging.

## 6. Save, reload and evaluate

`train.py` runs the complete experiment and saves all five fitted pipelines, audit manifests, CV tables, held-out results, predictions and plots. `model/metadata.json` records feature names, class mapping, package versions, selected parameters, split details and checksums.

`spambase/evaluation.py` supplies the same evaluator to training, the terminal CLI and the app. It calls `predict` and `predict_proba` on already fitted pipelines. AUC selects the probability column for class 1 using `model.classes_`; it does not use class predictions as ranking scores. Evaluation never calls `fit`.

`evaluate.py` prints the comparison for a labelled CSV and can select one model or export a CSV. `app.py` adds upload, model selection, all-six-metric display, reports, the all-model comparison and downloadable predictions. Every row of its model comparison uses the currently active CSV.

CSV validation requires the exact 57 input names and 0/1 `spam` target. Reordered columns are accepted and restored to the model's schema. Duplicate headers, missing/extra columns, malformed rows, nonnumeric values, missing values and infinity are rejected.

## 7. Connect observations to measured errors

Read the README tables alongside `results/evaluation_details.json`. Identify which model misses more spam and which flags more legitimate email. Models can share an accuracy while having different precision and recall because their false positives and false negatives differ.

SVM is the selected winner for this run based on training CV, and its held-out F1 is also highest. Logistic Regression has higher spam recall, while Naive Bayes detects still more spam but has many more false positives. These are measured tradeoffs on the fixed test split, not universal rankings.

If a later run has a different test leader from its CV-selected model, preserve the training-based selection rule and describe the test leader as an observation. Avoid changing seeds or parameters repeatedly in response to the test table. The historical, collection-specific dataset and lack of sender/campaign/time grouping limit generalisation beyond this experiment.

## 8. Reproduce and complete the external work

From this folder, run `.venv/bin/python train.py` to rebuild artifacts, `.venv/bin/python evaluate.py` to reload/evaluate them, and `.venv/bin/python -m streamlit run app.py` to launch the interface. Install development dependencies and run `python -m pytest -q` for automated checks.
