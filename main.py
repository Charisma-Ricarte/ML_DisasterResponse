from src.preprocess import load_and_preprocess
from src.model import get_model
from src.train_eval import train_model

# 1. Load and preprocess
train_enc, val_enc, train_labels, val_labels, num_labels = load_and_preprocess("data\tweets.csv")

# 2. Initialize model
model = get_model(num_labels)

# 3. Train model
trained_model = train_model(model, train_enc, train_labels, val_enc, val_labels, epochs=2, lr=2e-5)
