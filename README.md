# Comsys-hackathon-5 by Team Face Monk
## Task A
### Objective
This project uses a **transfer learning-based deep learning model** (ResNet50) to classify face images as either **male** or **female**.
### 🚀 Getting Started
#### 1. Install Dependencies
From your terminal or notebook cell:
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
###  Notes
The trained model expects input shape (224, 224, 3) and pixel values scaled to [0, 1].

## Task B
### Objective
The goal of this project is to match an input face image (or its distorted version) to its correct identity match.
If the input image matches **any image in the same folder**, it's considered a **match** (`label = 1`)
If the input image matches an image from **another person's folder**, it's a **non-match** (`label = 0`)

### 🚀 Getting Started
#### 1. Install Dependencies
Similar to Task A, open notebook cell
Make sure you have TensorFlow 2.x installed.
Tensorflow will not work with python version above 3.11. Please try to use python 3.10 to get best performance.
use:
```bash
!pip install tensorflow matplotlib scikit-learn
```
### Training
Run task_b.ipynb
Replace the dataset path with your local path

This trains the Siamese model and saves it as face_verifier.h5.

### Testing

Run task_b_test.ipynb

This evaluates the model on the validation dataset and:

1. Shows prediction samples with their similarity scores

2. Plots confusion matrix

3. Prints Accuracy, Precision, Recall, F1-Score
###  Notes
The trained model expects input shape (160, 160, 3).
Training time will depend on dataset size and hardware (GPU/CPU)

You can reduce training time by:

1. Limiting number of training pairs (e.g. 2000–3000)

2. Reducing number of epochs


## Run streamlit app for both task (Optional)
### ⚙️ Setup Instructions

#### 1️⃣ Create Virtual Environment

```bash
python -m venv myenv
```
#### Activate Environment
For windows:

```bash
myenv\Scripts\activate

```
For macOS/Linux
```bash
source myenv/bin/activate
```
####  Install Dependencies
Install all dependencies for each task as mentioned above along with Streamlit

#####  Run App
```bash
streamlit run app.py

```
This will open the app in your browser at http://localhost:8501.

