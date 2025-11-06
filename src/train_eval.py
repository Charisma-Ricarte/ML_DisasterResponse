import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def train_model(model, train_enc, train_labels, val_enc, val_labels, epochs=2, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], torch.tensor(train_labels))
    val_dataset   = TensorDataset(val_enc['input_ids'], val_enc['attention_mask'], torch.tensor(val_labels))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids, mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids, attention_mask=mask)
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                true.extend(labels.cpu().numpy())
        acc = accuracy_score(true, preds)
        val_accuracies.append(acc)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}")

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Training Progress")
    plt.show()

    return model
