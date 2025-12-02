import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_and_split_training_data(train_dir, img_size=(32, 32), test_size=0.1):
    train_data = []
    train_labels = []

    for folder in os.listdir(train_dir):
        if(folder == '.DS_Store'):
            continue
        folder_path = os.path.join(train_dir, folder)

        for image_file in os.listdir(folder_path):
            path = os.path.join(folder_path, image_file)
            image = Image.open(path).resize(img_size)
            image = np.array(image)

            train_data.append(image)
            train_labels.append(int(folder))

    train_data = np.array(train_data).astype("float32")  # ✅ NO normalization here
    train_labels = np.array(train_labels)

    X_train, X_val, y_train, y_val = train_test_split(
        train_data,
        train_labels,
        test_size=test_size,
        stratify=train_labels,
        random_state=30
    )

    return X_train, X_val, y_train, y_val

def load_test_data(data_path, test_csv, img_size=(32, 32)):
    test_data = []
    test_filenames = (data_path + test_csv.Path).tolist()

    for test_filename in test_filenames:
        image = Image.open(test_filename).resize(img_size)
        image = np.array(image)
        test_data.append(image)

    X_test = np.array(test_data).astype("float32")  # ✅ NO normalization here
    y_test = np.array(test_csv.ClassId.tolist())

    return X_test, y_test


