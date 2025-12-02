import matplotlib.pyplot as plt
import os
import cv2
from utils.constants import SIGN_CLASSES
import numpy as np

def show_meta_images(meta_files, images_per_page=18, rows=3, cols=6):
    num_pages = (len(meta_files) + images_per_page - 1) // images_per_page

    for page in range(num_pages):
        plt.figure(figsize=(12, 8))
        start = page * images_per_page
        end = start + images_per_page
        chunk = meta_files[start:end]

        for i, file in enumerate(chunk):
            class_id = int(os.path.splitext(os.path.basename(file))[0])

            plt.subplot(rows, cols, i + 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.title(SIGN_CLASSES[class_id], fontsize=7, wrap=True)

            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

        plt.tight_layout()
        plt.show()

def plot_training_curves(train_accs, val_accs, train_losses, val_losses):
    plt.figure()
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm):
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()



def show_test_predictions(X_test, y_test, predictions, max_show=30):
    plt.figure(figsize=(30, 30))
    max_show = min(max_show, len(X_test))

    for i in range(max_show):
        plt.subplot(10, 3, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        pred = predictions[i]
        gt = y_test[i]
        col = 'g' if pred == gt else 'r'

        plt.xlabel(
            f'Actual {SIGN_CLASSES[gt]} , Predicted {SIGN_CLASSES[pred]}',
            color=col,
            weight='bold'
        )

        plt.imshow(X_test[i].astype(np.uint8))

    plt.show()
