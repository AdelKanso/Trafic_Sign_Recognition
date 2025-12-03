import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from evaluation.visualization import show_final_comparison_window


def train(model, train_loader, val_loader,device, criterion, optimizer, epochs=30, patience=5):
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        correct, total, running_loss = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        correct, total, running_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images), labels)

                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                running_loss += loss.item()

        val_acc = correct / total
        val_loss = running_loss / len(val_loader)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_accs, val_accs, train_losses, val_losses


def train_and_compare(model, name,
                      train_loader, val_loader, test_loader,
                      X_test_t, y_test, device,
                      return_curves=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    results = train(
        model, train_loader, val_loader,
        device, criterion, optimizer, epochs=2
    )

    if return_curves:
        train_accs, val_accs, train_losses, val_losses = results
    else:
        train_accs = val_accs = train_losses = val_losses = None

    model.eval()
    all_preds = []

    with torch.no_grad():
        for (images,) in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu().numpy())

    predictions = np.hstack(all_preds)

    show_final_comparison_window(
        y_true=y_test,
        y_pred=predictions,
        model=model,
        X_test_t=X_test_t,
        device=device,
        dataset_name="GTSRB",
        model_name=name
    )

    return train_accs, val_accs, train_losses, val_losses

