from torchvision.datasets import ImageFolder

DATASET_FLAG = 0 # 0: GTSRB, 1: Belgium

TRAIN_MODEL = 0 # 0: Our Model, 1: ResNet18, 2: MobileNetV2


if DATASET_FLAG == 0:
    DATA_PATH = "dataset/gtsrb_dataset/"
    META_DIR = "Meta/"
    TRAIN_DIR = DATA_PATH + "Train/"
    TEST_DIR = DATA_PATH + "Test/"

    SIGN_CLASSES = {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing for vehicles over 3.5t",
        11: "Right-of-way at the next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles over 3.5t prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve to the left",
        20: "Dangerous curve to the right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End of all speed and passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing for vehicles over 3.5t"
    }

    NUM_CLASSES = 43

elif DATASET_FLAG == 1:
    DATA_PATH = "dataset/belgium/"
    META_DIR = None
    TRAIN_DIR = DATA_PATH + "Training/"
    TEST_DIR = DATA_PATH + "Testing/"

    _tmp_dataset = ImageFolder(TRAIN_DIR)
    SIGN_CLASSES = {v: k for k, v in _tmp_dataset.class_to_idx.items()}
    NUM_CLASSES = len(SIGN_CLASSES)
