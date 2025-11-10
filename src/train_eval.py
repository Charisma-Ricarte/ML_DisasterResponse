import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(model, train_enc, val_enc, train_labels, val_labels, epochs=2, batch_size=16, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Prepare data loaders
    train_data = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels))
    val_data = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(val_labels))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            b_input_ids, b_attn_mask, b_labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids, b_attn_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == b_labels).sum().item()
                total += b_labels.size(0)

        val_acc = correct / total
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Plot loss curve
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss_curve.png")
    plt.close()

    return train_losses, val_accs

