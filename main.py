import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from dataset.augmented_dataset import AugmentedDataset
from dataset.loaders import load_and_split_training_data, load_test_data
from evaluation.visualization import (
    plot_class_accuracy,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_single_image_probabilities,
    plot_training_curves,
    show_meta_images,
    show_test_predictions,
)
from model.alternatives import build_mobile_net, build_resnet18
from model.cnn import CNN
from training.training import train_and_compare
from utils.constants import DATA_PATH, SIGN_CLASSES, TRAIN_DIR, TRAIN_MODEL
from utils.file_loader import FilePath


def main():
    files = FilePath()
    files.load_all()
    show_meta_images(files.meta_files)

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
    
    if(TRAIN_MODEL == 0):
        model = CNN(num_classes=43).to(device)

        train_accs, val_accs, train_losses, val_losses = train_and_compare(
            model=model,
            name="Ours",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            X_test_t=X_test_t,
            y_test=y_test,
            device=device,
            return_curves=True   
        )
        

        plot_training_curves(train_accs, val_accs, train_losses, val_losses)

        model.eval()
        
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for (images,) in test_loader:
                images = images.to(device)
                outputs = model(images)
                
                probs = torch.softmax(outputs, dim=1)   # probabilities
                preds = probs.argmax(1)                # class labels
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())   


        all_probs = np.vstack(all_probs)
        predictions = np.hstack(all_preds)

        
        plot_pr_curve(
            y_true=y_test,
            y_probs=all_probs,
            num_classes=43
        )


        print("Test Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(
            y_test,
            predictions,
            target_names=[SIGN_CLASSES[i] for i in sorted(SIGN_CLASSES)]
        ))


        cm = confusion_matrix(y_test, predictions)
        plot_confusion_matrix(cm)

        show_test_predictions(X_test, y_test, predictions)


        plot_class_accuracy(
            y_true=y_test,
            y_pred=predictions,
            class_names=[SIGN_CLASSES[i] for i in sorted(SIGN_CLASSES)]
        )

        # Take one test image (for example: index 0)
        image_tensor = X_test_t[0].to(device)

        plot_single_image_probabilities(
            model,
            image_tensor,
            [SIGN_CLASSES[i] for i in sorted(SIGN_CLASSES)]
        )


    if(TRAIN_MODEL == 1):
        resnet_model = build_resnet18(num_classes=43).to(device)

        train_and_compare(
            model=resnet_model,
            name="ResNet18",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            X_test_t=X_test_t,
            y_test=y_test,
            device=device,
            return_curves=False
        )
        
    if(TRAIN_MODEL == 2):
        mobile_net_model = build_mobile_net(num_classes=43).to(device)
        train_and_compare(
            model=mobile_net_model,
            name="MobileNetV2",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            X_test_t=X_test_t,
            y_test=y_test,
            device=device,
            return_curves=False
        )
      
        
    input("\nPress ENTER to close")

if __name__ == '__main__':
    main()
