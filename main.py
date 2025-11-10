from src.preprocess import load_and_preprocess
from src.model import create_model
from src.train_eval import train_model
from src.utils import plot_metrics

if __name__ == "__main__":
    # Load dataset and preprocess
    train_enc, val_enc, train_labels, val_labels, num_labels = load_and_preprocess("data/tweets.csv")

    # Create and train model
    model = create_model(num_labels)
    train_losses, val_accs = train_model(model, train_enc, val_enc, train_labels, val_labels)

    # Plot summary metrics
    plot_metrics(train_losses, val_accs)

    print("Training complete. Metrics and plots saved.")
