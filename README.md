# Comsys-hackathon-5 by Team Face Monk
## Task A
### Objective
This project uses a **transfer learning-based deep learning model** (ResNet50) to classify face images as either **male** or **female**.
### 🚀 Getting Started
#### 1. Install Dependencies
rom your terminal or notebook cell:
```bash
!pip install tensorflow matplotlib scikit-learn pillow

```
Run all cells
Go to Kernel → Restart & Run All, or run cell by cell.

Make sure:

The dataset folders are correctly referenced:
```python
train_path = "/kaggle/input/comsys/Comys_Hackathon5/Task_A/train"
val_path   = "/kaggle/input/comsys/Comys_Hackathon5/Task_A/val"

```
Replace the paths with your local dataset paths

After training:

The model will be saved as gender_classification.h5

Training accuracy/loss and validation accuracy/loss will be plotted

Optional: Enable GPU (if local)
Ensure TensorFlow detects GPU:
```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

```

## Task B
### Objective
The goal of this project is to match an input face image (or its distorted version) to its correct identity match.
If the input image matches **any image in the same folder**, it's considered a **match** (`label = 1`)
If the input image matches an image from **another person's folder**, it's a **non-match** (`label = 0`)

### 🚀 Getting Started
#### 1. Install Dependencies
Make sure you have TensorFlow 2.x installed.
Tensorflow will not work with python version above 3.11. Please try to use python 3.10 to get best performance.
use:
```bash
pip install tensorflow
```
### Training
Run task_b.ipynb

This trains the Siamese model and saves it as face_verifier.h5.

### Testing

Run task_b_test.ipynb

This evaluates the model on the validation dataset and:

1. Shows prediction samples with their similarity scores

2. Plots confusion matrix

3. Prints Accuracy, Precision, Recall, F1-Score
