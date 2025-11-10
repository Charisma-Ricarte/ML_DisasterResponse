import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_and_preprocess(path):
    abs_path = os.path.abspath(path)
    print("Reading file from:", abs_path)

    df = pd.read_csv(abs_path)
    print("Dataset shape:", df.shape)
    print(df.head())

    # Simple EDA plot for Update 1
    df["target"].value_counts().plot(
        kind="bar",
        title="Target Distribution (0 = Non-disaster, 1 = Disaster)"
    )
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png")
    plt.close()

    # Tokenize text for model input
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    texts = df["text"].astype(str).tolist()
    labels = df["target"].astype(int).tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    train_idx, val_idx, _, _ = train_test_split(
        range(len(labels)), labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_enc = {k: v[train_idx] for k, v in encodings.items()}
    val_enc = {k: v[val_idx] for k, v in encodings.items()}
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    num_labels = len(set(labels))

    return train_enc, val_enc, train_labels, val_labels, num_labels
