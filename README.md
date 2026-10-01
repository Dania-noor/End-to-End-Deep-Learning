# End-to-End Deep Learning Pipeline

## 📌 Project Overview

This project demonstrates an end-to-end **Deep Learning Pipeline** for binary image classification using a Cats and Dogs dataset.

The project implements and compares five different deep learning approaches:

* Artificial Neural Network (ANN)
* Convolutional Neural Network (CNN)
* Recurrent Neural Network (RNN)
* Long Short-Term Memory (LSTM)
* Transfer Learning

The objective is to understand how different deep learning architectures perform on an image classification problem and to evaluate their performance using multiple metrics.

---

## 🎯 Objectives

The main objectives of this project are:

* Build an end-to-end deep learning pipeline
* Load and organize image data
* Perform image preprocessing
* Normalize image pixel values
* Split the dataset into training and testing sets
* Implement ANN, CNN, RNN, and LSTM models
* Implement Transfer Learning using MobileNetV2
* Train and validate all models
* Evaluate model performance
* Compare different architectures
* Visualize model performance
* Save trained models
* Document experimental results

---

## 📊 Dataset

The project uses a **Cats and Dogs image dataset** containing 1,000 images.

The dataset is balanced between the two classes.

### Dataset Information

| Property     |                 Value |
| ------------ | --------------------: |
| Total Images |                 1,000 |
| Cat Images   |                   500 |
| Dog Images   |                   500 |
| Image Size   |             128 × 128 |
| Channels     |               3 (RGB) |
| Problem Type | Binary Classification |

### Classes

| Label | Class |
| ----: | ----- |
|     0 | Cat   |
|     1 | Dog   |

The images are organized into separate folders for cats and dogs.

---

## 🔄 Deep Learning Pipeline

The complete workflow is:

```text
Dataset
   ↓
Data Ingestion
   ↓
Image Loading
   ↓
Image Resizing
   ↓
Pixel Normalization
   ↓
Train/Test Split
   ↓
Model Building
   ↓
Model Training
   ↓
Validation
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Visualization
   ↓
Best Model Selection
```

---

## 📥 1. Data Ingestion

The data ingestion module scans the Cats and Dogs dataset and creates a structured DataFrame containing:

* Image filename
* Image filepath
* Class label

The labels are assigned as:

```text
Cat → 0
Dog → 1
```

The final dataset contains:

```text
Total images: 1000

Cat: 500
Dog: 500
```

---

## 🧹 2. Data Preprocessing

The images are loaded and prepared for deep learning models.

The preprocessing steps include:

* Loading images
* Resizing images to 128 × 128 pixels
* Converting images to RGB format
* Converting images into numerical arrays
* Normalizing pixel values
* Splitting the dataset into training and testing sets

### Dataset Split

```text
Total Dataset: 1000
       │
       ├── Training: 800
       │
       └── Testing: 200
```

### Final Shapes

```text
X shape:       (1000, 128, 128, 3)
y shape:       (1000,)

X_train:       (800, 128, 128, 3)
y_train:       (800,)

X_test:        (200, 128, 128, 3)
y_test:        (200,)
```

Pixel values were normalized to the range:

```text
Minimum: 0.0
Maximum: 1.0
```

---

# 🤖 Deep Learning Models

Five different architectures were implemented and compared.

---

## 1. Artificial Neural Network (ANN)

The ANN model processes the image by first flattening the image pixels into a one-dimensional vector.

### Architecture

```text
Input Image
     ↓
Flatten
     ↓
Dense (128)
     ↓
Dropout
     ↓
Dense (64)
     ↓
Dropout
     ↓
Dense (1)
     ↓
Binary Classification
```

The ANN achieved limited performance because flattening an image removes much of the spatial information that is important for image classification.

---

## 2. Convolutional Neural Network (CNN)

The CNN model is designed specifically for image data.

### Architecture

```text
Input Image
     ↓
Conv2D (32)
     ↓
MaxPooling
     ↓
Conv2D (64)
     ↓
MaxPooling
     ↓
Conv2D (128)
     ↓
MaxPooling
     ↓
Flatten
     ↓
Dense (128)
     ↓
Dropout
     ↓
Output
```

CNNs can learn spatial patterns and visual features from images, making them more suitable for image classification than a basic ANN.

---

## 3. Recurrent Neural Network (RNN)

An RNN was implemented to demonstrate how recurrent architectures can be applied to image data by representing image information as sequences.

### Architecture

```text
Image Sequence
     ↓
SimpleRNN (64)
     ↓
Dense (64)
     ↓
Dropout
     ↓
Output
```

The RNN achieved moderate performance but was not as effective as the CNN or Transfer Learning model for this image classification task.

---

## 4. Long Short-Term Memory (LSTM)

LSTM is an advanced recurrent neural network designed to handle sequential dependencies.

### Architecture

```text
Image Sequence
     ↓
LSTM (64)
     ↓
Dense (64)
     ↓
Dropout
     ↓
Output
```

The LSTM model was also evaluated to compare its performance with the basic RNN.

---

## 5. Transfer Learning

Transfer Learning was implemented using **MobileNetV2**.

MobileNetV2 is a pretrained convolutional neural network that was used as a feature extractor.

The pretrained base was kept frozen while a new classification head was added.

### Architecture

```text
Input Image
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense (128)
     ↓
Dropout
     ↓
Output
```

Transfer Learning achieved the best performance among all implemented models.

---

# 🏋️ Model Training

All five models were trained for 10 epochs using a training and validation split.

The training process monitored:

* Training accuracy
* Training loss
* Validation accuracy
* Validation loss

The trained models were saved in Keras format.

---

# 📊 Model Evaluation

The models were evaluated using:

* Test Loss
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

These metrics provide a more complete understanding of model performance than accuracy alone.

---

# 🏆 Final Results

The final test results obtained from the project are:

| Model                 |  Test Loss |   Accuracy |  Precision |     Recall |   F1 Score |
| --------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| ANN                   |     0.6932 |     50.00% |      0.00% |      0.00% |      0.00% |
| CNN                   |     0.6703 |     62.00% |     58.33% |     84.00% |     68.85% |
| RNN                   |     0.6752 |     60.50% |     58.27% |     74.00% |     65.20% |
| LSTM                  |     0.6732 |     55.50% |     54.55% |     66.00% |     59.73% |
| **Transfer Learning** | **0.0526** | **98.50%** | **98.02%** | **99.00%** | **98.51%** |

---

# 🥇 Best Model

The best-performing model was:

## Transfer Learning — MobileNetV2

### Performance

* **Accuracy:** 98.50%
* **Precision:** 98.02%
* **Recall:** 99.00%
* **F1 Score:** 98.51%
* **Test Loss:** 0.0526

### Confusion Matrix

```text
[[98   2]
 [ 1  99]]
```

This means:

* 98 cats were correctly classified
* 2 cats were incorrectly classified
* 99 dogs were correctly classified
* 1 dog was incorrectly classified

The results show that Transfer Learning was significantly more effective than the basic ANN, CNN, RNN, and LSTM models for this dataset.

---

# 📈 Model Comparison

The model comparison showed the following accuracy:

```text
ANN                50.00%
CNN                62.00%
RNN                60.50%
LSTM               55.50%
Transfer Learning  98.50%
```

Transfer Learning produced a substantial improvement because MobileNetV2 already contains learned visual features from large-scale image training.

---

# 📁 Project Structure

```text
02_End_to_End_Deep_Learning/
│
├── data/
│   ├── cats_set/
│   └── dogs_set/
│
├── models/
│   ├── ANN.keras
│   ├── CNN.keras
│   ├── RNN.keras
│   ├── LSTM.keras
│   └── Transfer_Learning.keras
│
├── results/
│   └── model_comparison.csv
│
├── graphs/
│
├── S1_DataIngestion.py
├── S2_PreProcessing.py
├── S3_ModelBuilding.py
├── S4_ModelTraining.py
├── S5_ModelEvaluation.py
├── S6_Visualization.py
├── Main.py
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Pillow (PIL)
* PyCharm

---

# 📦 Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn pillow
```

Or install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Run the modules in the following order:

```bash
python S1_DataIngestion.py
python S2_PreProcessing.py
python S3_ModelBuilding.py
python S4_ModelTraining.py
python S5_ModelEvaluation.py
python S6_Visualization.py
```

The complete pipeline can also be executed using:

```bash
python Main.py
```

---

# 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

* End-to-end Deep Learning pipelines
* Image data preprocessing
* Image normalization
* Binary image classification
* Artificial Neural Networks
* Convolutional Neural Networks
* Recurrent Neural Networks
* LSTM networks
* Transfer Learning
* MobileNetV2
* Model training and validation
* Classification metrics
* Confusion matrices
* Model comparison
* Visualization of experimental results
* Saving and reusing trained models

---

# 🚀 Future Improvements

Possible improvements include:

* Increasing the size of the training dataset
* Applying data augmentation
* Fine-tuning the pretrained MobileNetV2 layers
* Experimenting with other pretrained architectures
* Using early stopping and learning-rate scheduling
* Performing more extensive hyperparameter tuning
* Deploying the best model as an image classification application

---

## 👩‍💻 Author

**Dania Noor**

Computer Science Undergraduate
NED University of Engineering and Technology
