import re
from pathlib import Path

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


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

    label_encoder = LabelEncoder()
    work["label_id"] = label_encoder.fit_transform(work["label"])
    label_names = list(label_encoder.classes_)

    train_df, test_df = train_test_split(
        work[["clean_text", "label_id"]],
        test_size=0.2,
        random_state=42,
        stratify=work["label_id"]
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names)
    )

    training_args = TrainingArguments(
        output_dir="results/bert_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="results/bert_logs",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training BERT...")
    trainer.train()

    print("Evaluating BERT...")
    results = trainer.evaluate()
    pd.DataFrame([results]).to_csv(TABLES_DIR / "bert_results.csv", index=False)

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(TABLES_DIR / "bert_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("BERT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bert_confusion_matrix.png", dpi=300)
    plt.close()

    print("Done.")
    print("BERT results saved in results/tables and results/figures.")
    print(results)


if __name__ == "__main__":
    main()
