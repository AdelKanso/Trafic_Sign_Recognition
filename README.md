# Traffic Sign Detection (GTSRB)

A machine learning project for detecting and classifying German traffic signs using the GTSRB (German Traffic Sign Recognition Benchmark) dataset. This project uses a **Convolutional Neural Network (CNN)** trained on augmented traffic sign images to achieve high accuracy classification.


## Prerequisites

- Python 3.7 or higher
- macOS, Linux, or Windows
- ~0.5 GB of free disk space (for the dataset)
- pip (Python package manager)
- GTSRB https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- BelgiumTS https://btsd.ethz.ch/shareddata/

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

## License

This project uses the GTSRB dataset from Kaggle. Please refer to the dataset's license terms.
This project uses the BelgiumTS dataset from btsd. Please refer to the dataset's license terms.

## References

- [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeow/gtsrb-german-traffic-sign)
- [BelgiumTS Dataset](https://btsd.ethz.ch/shareddata/)
- [Original GTSRB Paper](http://benchmark.ini.rub.de/)
