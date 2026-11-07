"""One shared schema for data preparation, trained pipelines and the app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "spam"
SEED = 42
TEST_SIZE = 0.20
DATASET_URL = "https://archive.ics.uci.edu/dataset/94/spambase"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
FEATURE_COLUMNS = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total",
]
MODEL_SPECS = [
    {"slug": "logistic_regression", "name": "Logistic Regression"},
    {"slug": "decision_tree", "name": "Decision Tree"},
    {"slug": "knn", "name": "KNN"},
    {"slug": "naive_bayes", "name": "Naive Bayes"},
    {"slug": "svm", "name": "Support Vector Machine"},
]
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
