import re
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

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
        raise FileNotFoundError("No CSV files found inside data/raw.")

    frames = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            if not df.empty:
                df["source_file"] = str(file)
                frames.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if not frames:
        raise ValueError("No readable CSV files were found.")

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

    if any(word in label for word in ["malware", "hack", "exploit", "ransomware", "botnet", "ddos", "rat", "crypter", "keylogger"]):
        return "Malware/Hacking Tools"

    if any(word in label for word in ["fake", "counterfeit", "passport", "document", "id", "forg"]):
        return "Counterfeit/Forgeries"

    return "Digital/Other"


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_name", type=str, default="distilbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    print("Loading DWData...")
    df = load_dwdata()

    print("Dataset shape:", df.shape)

    text_col = find_column(df, ["Product Name", "name", "title", "description", "product"])
    label_col = find_column(df, ["Category", "category", "cat", "type"])

    if text_col is None or label_col is None:
        raise ValueError("Could not identify text or label columns.")

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = TextClassificationDataset(
        train_df["clean_text"],
        train_df["label_id"],
        tokenizer,
        max_length=args.max_length
    )

    test_dataset = TextClassificationDataset(
        test_df["clean_text"],
        test_df["label_id"],
        tokenizer,
        max_length=args.max_length
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names)
    )

    training_args = TrainingArguments(
        output_dir=f"results/{args.output_name}_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f"Training {args.model_name}...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()

    pd.DataFrame([results]).to_csv(
        TABLES_DIR / f"{args.output_name}_results.csv",
        index=False
    )

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

    pd.DataFrame(report).transpose().to_csv(
        TABLES_DIR / f"{args.output_name}_classification_report.csv"
    )

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
    plt.title(f"{args.output_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{args.output_name}_confusion_matrix.png", dpi=300)
    plt.close()

    print("Done.")
    print(results)


if __name__ == "__main__":
    main()
