import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_accs):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics_summary.png")
    plt.close()
