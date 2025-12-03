import glob
import os
import pandas as pd
from utils.constants import DATASET_FLAG, DATA_PATH, TRAIN_DIR, META_DIR


class FilePath:
    def __init__(self):
        self.meta_files = None
        self.meta_csv = None
        self.train_dir = None
        self.test_csv = None

    def load_all(self):
        if DATASET_FLAG == 0:
            self.meta_files = glob.glob(os.path.join(DATA_PATH, META_DIR, "*.*"))
            self.meta_csv = pd.read_csv(os.path.join(DATA_PATH, "Meta.csv"))
            self.train_dir = os.listdir(os.path.join(DATA_PATH, "Train"))
            self.test_csv = pd.read_csv(os.path.join(DATA_PATH, "Test.csv"))
        else:
            self.train_dir = os.listdir(TRAIN_DIR)
