import io
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from utils.common import softmax_with_temperature
from utils.constants import SIGN_CLASSES


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

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.yticks(range(len(class_names)), class_names, fontsize=8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.show()



def show_test_predictions(X_test, y_test, predictions):
    images_per_page = 12
    rows, cols = 4, 3 
    max_pages = 4  

    total = min(len(X_test), images_per_page * max_pages)
    num_pages = (total + images_per_page - 1) // images_per_page

    for page in range(num_pages):
        plt.figure(figsize=(18, 16))

        start = page * images_per_page
        end = start + images_per_page

        chunk_X = X_test[start:end]
        chunk_y = y_test[start:end]
        chunk_pred = predictions[start:end]

        for i in range(len(chunk_X)):
            plt.subplot(rows, cols, i + 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            pred = chunk_pred[i]
            gt = chunk_y[i]
            col = 'g' if pred == gt else 'r'

            plt.xlabel(
                f'Actual {SIGN_CLASSES[gt]} | Pred {SIGN_CLASSES[pred]}',
                color=col,
                weight='bold',
                fontsize=9
            )

            plt.imshow(chunk_X[i].astype(np.uint8))

        plt.suptitle(
            f"Test Predictions — Page {page + 1}/{num_pages}",
            fontsize=18, weight="bold"
        )

        plt.tight_layout()
        plt.show()


def plot_pr_curve(y_true, y_probs, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    precision, recall, _ = precision_recall_curve(
        y_true_bin.ravel(),
        y_probs.ravel()
    )

    ap_score = average_precision_score(
        y_true_bin,
        y_probs,
        average="micro"
    )

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Micro PR Curve (AP = {ap_score:.4f})")
    plt.grid()

    plt.show(block=True)


def plot_class_accuracy(y_true, y_pred, class_names):
    num_classes = len(class_names)
    class_acc = []

    for i in range(num_classes):
        idx = (y_true == i)
        if np.sum(idx) == 0:
            acc = 0
        else:
            acc = np.mean(y_pred[idx] == y_true[idx]) * 100 

        class_acc.append(acc)

    plt.figure(figsize=(16, 6))
    plt.bar(range(num_classes), class_acc)
    plt.xticks(range(num_classes), class_names, rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Classes")
    plt.title("Per-Class Accuracy (%)")
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()


def plot_single_image_probabilities(model, image_tensor, class_names):
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probs = softmax_with_temperature(logits, T=3.0).cpu().numpy()[0] * 100
    for c, p in zip(class_names, probs):
        print(c, p)
    plt.figure(figsize=(16, 6))
    plt.bar(range(len(class_names)), probs)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.ylabel("Probability (%)")
    plt.xlabel("Classes")
    plt.title("Class Probability Distribution for One Traffic Sign")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def show_final_comparison_window(
    y_true,
    y_pred,
    model,
    X_test_t,
    device,
    dataset_name="GTSRB",
    model_name="Ours"
):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_mb = buffer.getbuffer().nbytes / (1024 ** 2)

    model.eval()
    x = X_test_t[:200].to(device)  # test on 200 images only (fast)

    with torch.no_grad():
        for _ in range(5):   
            _ = model(x[:16])

        t0 = time.time()
        _ = model(x)
        t1 = time.time()

    avg_time_ms = (t1 - t0) / x.shape[0] * 1000

    df = pd.DataFrame({
        "Dataset": [dataset_name],
        "Model": [model_name],
        "Accuracy (%)": [round(acc * 100, 2)],
        "F1-score (%)": [round(f1 * 100, 2)],
        "Model Size (MB)": [round(model_size_mb, 2)],
        "Inference Time (ms/img)": [round(avg_time_ms, 2)]
    })

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    plt.title("Comparison Table", fontsize=14, pad=10)
    plt.tight_layout()
    plt.show()
