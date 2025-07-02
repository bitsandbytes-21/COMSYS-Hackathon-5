# Comsys-hackathon-5 by Team Face Monk
## Task A
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

```bash
run task_b.ipynb
```
This trains the Siamese model and saves it as face_verifier.h5.

### Testing
```bash
run task_b_test.ipynb
```
This evaluates the model on the validation dataset and:

1. Shows prediction samples with their similarity scores

2. Plots confusion matrix

3. Prints Accuracy, Precision, Recall, F1-Score
