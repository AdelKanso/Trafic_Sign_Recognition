import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from dataset.augmented_dataset import AugmentedDataset
from dataset.loaders import load_and_split_training_data, load_test_data
from evaluation.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    show_meta_images,
    show_test_predictions,
)
from neural_network.cnn import CNN
from training.training import train
from utils.constants import DATA_PATH, SIGN_CLASSES, TRAIN_DIR
from utils.file_loader import FilePath


def main():
    files = FilePath()
    files.load_all()
    # show_meta_images(files.meta_files)

    X_train, X_val, y_train, y_val = load_and_split_training_data(TRAIN_DIR)
    X_test, y_test = load_test_data(DATA_PATH, files.test_csv)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = AugmentedDataset(X_train, y_train, train_transform)
    val_dataset   = AugmentedDataset(X_val, y_val, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)

    X_test_t = torch.tensor(X_test).permute(0, 3, 1, 2) / 255.0
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=64)

    model = CNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_accs, val_accs, train_losses, val_losses = train(
        model, train_loader, val_loader,device, criterion, optimizer, epochs=2
    )
    

    plot_training_curves(train_accs, val_accs, train_losses, val_losses)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for (images,) in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    predictions = np.array(all_preds)

    print("Test Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(
        y_test,
        predictions,
        target_names=[SIGN_CLASSES[i] for i in sorted(SIGN_CLASSES)]
    ))


    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm)

    show_test_predictions(X_test, y_test, predictions)


if __name__ == '__main__':
    main()
