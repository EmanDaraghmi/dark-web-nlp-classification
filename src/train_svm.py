import re
from pathlib import Path

import pandas as pd
import joblib
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


RAW_DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\-_/$£€]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dwdata():
    csv_files = list(RAW_DATA_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found inside data/raw. "
            "Please unzip DWData-master.zip into data/raw/"
        )

    frames = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            df["source_file"] = str(file)
            frames.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")

    return pd.concat(frames, ignore_index=True)


def find_column(df, candidates):
    lower_cols = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_cols:
            return lower_cols[candidate.lower()]

    return None


def map_category(label):
    label = str(label).lower()

    if any(word in label for word in ["drug", "cannabis", "weed", "cocaine", "opioid", "mdma"]):
        return "Drugs"

    if any(word in label for word in ["fraud", "card", "bank", "account", "stolen", "fullz", "paypal"]):
        return "Fraud/Stolen Data"

    if any(word in label for word in ["malware", "hack", "exploit", "ransomware", "botnet", "ddos"]):
        return "Malware/Hacking Tools"

    if any(word in label for word in ["fake", "counterfeit", "passport", "document", "id", "forg"]):
        return "Counterfeit/Forgeries"

    return "Digital/Other"


def main():
    print("Loading DWData...")
    df = load_dwdata()

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    text_col = find_column(df, ["Product Name", "name", "title", "description", "product"])
    label_col = find_column(df, ["Category", "category", "cat", "type"])

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not automatically identify text or label columns. "
            "Please inspect the printed columns and edit the script manually."
        )

    print("Selected text column:", text_col)
    print("Selected label column:", label_col)

    work = df[[text_col, label_col]].copy()
    work.columns = ["text", "raw_label"]

    work = work.dropna(subset=["text", "raw_label"])
    work["clean_text"] = work["text"].apply(clean_text)
    work["label"] = work["raw_label"].apply(map_category)

    work = work[work["clean_text"].str.len() > 0]
    work = work.drop_duplicates(subset=["clean_text", "label"])

    print("After cleaning:", work.shape)
    print("Class distribution:")
    print(work["label"].value_counts())

    work[["text", "clean_text", "raw_label", "label"]].head(50).to_csv(
        TABLES_DIR / "sample_cleaned_rows.csv",
        index=False
    )

    work["label"].value_counts().to_csv(TABLES_DIR / "class_distribution.csv")

    X = work["clean_text"]
    y = work["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("svm", LinearSVC())
    ])

    print("Training TF-IDF + Linear SVM...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(TABLES_DIR / "svm_classification_report.csv")

    results = {
        "model": "TF-IDF + Linear SVM",
        "accuracy": accuracy,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"]
    }

    pd.DataFrame([results]).to_csv(TABLES_DIR / "svm_results.csv", index=False)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("TF-IDF + Linear SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "svm_confusion_matrix.png", dpi=300)
    plt.close()

    joblib.dump(model, RESULTS_DIR / "svm_tfidf_model.joblib")

    print("Done.")
    print("Results saved in results/tables and results/figures.")


if __name__ == "__main__":
    main()
