import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_and_preprocess(path):
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print(df.head())

    # Handle missing values
    df.dropna(subset=['text'], inplace=True)

    # Visualize label distribution
    plt.figure(figsize=(6,4))
    df['label'].value_counts().plot(kind='bar', title='Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Split train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_enc = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt', max_length=64)
    val_enc = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt', max_length=64)

    return train_enc, val_enc, train_labels, val_labels, len(set(df['label']))
