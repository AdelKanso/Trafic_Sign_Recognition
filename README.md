# Traffic Sign Detection (GTSRB)

A machine learning project for detecting and classifying German traffic signs using the GTSRB (German Traffic Sign Recognition Benchmark) dataset. This project uses a **Convolutional Neural Network (CNN)** trained on augmented traffic sign images to achieve high accuracy classification. The trained model is exported as a PyTorch `.pth` file and used for real-time video detection.


## Prerequisites

- Python 3.7 or higher
- macOS, Linux, or Windows
- ~0.5 GB of free disk space (for the dataset)
- pip (Python package manager)
- dataset https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install numpy pandas opencv-python Pillow matplotlib scikit-learn torch torchvision scipy
```

## Dataset

The project uses the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset from Kaggle.

### Download Instructions:
1. Visit: [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeow/gtsrb-german-traffic-sign)
2. Download the dataset
3. Extract and place in `data/` folder:
   - `data/Train/` (43 class folders with training images)
   - `data/Test/` (test images and ground truth CSV)
   - `data/Meta.csv` (class metadata)
4. place it under dataset/ directly so it becomes dataset/gtsrb_dataset

## Usage

### Step 1: Train the Model

```bash
python main.py
```

This will:
- Load and preprocess training data from `data/Train/`
- Apply **data augmentation**
- Train the **CNN model** on the augmented dataset
- Evaluate on test set and generate performance metrics
- Export the trained model as `traffic_sign_model.pth`

### Step 2: Test on Video

```bash
python test_video.py
```

This will:
- Load the pre-trained model from `traffic_sign_model.pth`
- Process your video file frame-by-frame
- Detect traffic signs in real-time
- Display predictions and confidence scores
- Generate an output video with annotated detections

### Example Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Download and extract GTSRB dataset to data/ folder

# 3. Train model (generates traffic_sign_model.pth)
python main.py

# 4. Test on video
python test_video.py
```

## Dataset Overview

The GTSRB dataset contains:
- **43 traffic sign classes** (labeled 0–42)
- **39,209 training images** across 43 folders in `data/Train/`
- **12,630 test images** in `data/Test/`
- Image sizes: 32×32 to 224×224 pixels (variable)
- Includes real-world conditions: lighting variations, rain, snow, etc.

Each class folder (e.g., `data/Train/0/`, `data/Train/1/`) contains images for a specific traffic sign type.

## License

This project uses the GTSRB dataset from Kaggle. Please refer to the dataset's license terms.

## References

- [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeow/gtsrb-german-traffic-sign)
- [Original GTSRB Paper](http://benchmark.ini.rub.de/)

- [Video](https://youtube.com/shorts/60PevMQFm4g?si=LhV9TPgDfObSE5_x)
