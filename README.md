# DeepHAR: CNN & BiLSTM Based Human Activity Recognition

This repository presents **DeepHAR**, a comprehensive comparison of **classical machine learning** and **deep learning** approaches for **Human Activity Recognition (HAR)** using the **UCI HAR Dataset**. The dataset contains smartphone accelerometer and gyroscope time-series signals from 30 participants performing six daily activities.

DeepHAR implements:
* Random Forest
* SVM
* XGBoost
* 1D CNN
* BiLSTM

The **1D CNN achieves the highest accuracy**, demonstrating the strength of deep temporal feature extraction.

---

## 📁 Project Structure

```
Activity-Recognition/
│
├── data/                         # UCI HAR dataset
├── models/                       # ML & DL model scripts
│   ├── rf_model.py
│   ├── svm_model.py
│   ├── xgboost_model.py
│   ├── cnn.py
│   └── bilstm_model.py
│
├── notebooks/                    # Jupyter notebooks for ML/DL experiments
├── utils/
│   └── evaluate.py               # Accuracy, CM, and reports
│
├── results/                      # Generated plots & logs
├── main.py
├── requirements.txt
└── README.md
```

---

##  Models Implemented

### Classical Machine Learning

* Random Forest
* Support Vector Machine
* XGBoost

### Deep Learning

* 1D Convolutional Neural Network (CNN)
* Bidirectional LSTM (BiLSTM)

All classical models use the provided **561 engineered features**, while deep networks operate on reshaped time-series windows.

---

##  Model Accuracy Comparison

| Model         | Accuracy   |
| ------------- | ---------- |
| **CNN1D**     | **0.9576** |
| SVM           | 0.9505     |
| Random Forest | 0.9413     |
| XGBoost       | 0.9325     |
| BiLSTM        | 0.8351     |

---

##  CNN1D Architecture

The CNN1D architecture used in this project consists of:

* **Conv1D (kernel size = 3) → ReLU**
* **MaxPooling1D (pool size = 2)**
* **Conv1D + Dropout(0.5)**
* **Flatten**
* **Dense + Softmax (6 activity classes)**

This lightweight architecture effectively captures temporal patterns in sensor data.

---

##  CNN Architecture Diagram

![CNN Architecture](cnn_architecture.png)

---

##  Dataset Information

* 30 subjects performing **6 activities**
* Sampling frequency: **50 Hz**
* Window length: **128 samples**
* 561 engineered features included
* Standard **70/30 train–test split** provided

Activities:

* Walking
* Walking Up
* Walking Down
* Sitting
* Standing
* Laying

---

##  Running the Models

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Classical ML Models

```bash
python models/rf_model.py
python models/svm_model.py
python models/xgboost_model.py
```

### Run Deep Learning Models

Run the notebooks in the **notebooks/** directory.

---

##  Confusion Matrices & Metrics

All evaluation metrics (accuracy, classification reports, and confusion matrices) are saved in the **results/** folder.

The evaluation utility automatically generates:

* Accuracy score
* Per-class precision, recall, F1
* Heatmap confusion matrix

---

##  Future Enhancements

* Hybrid CNN–LSTM architecture
* Train on raw inertial signals instead of engineered features
* Convert trained models to **TensorFlow Lite** for mobile deployment
* Real-time HAR via smartphone sensors
* Automatic hyperparameter search (Optuna / Ray Tune)

---

##  License

MIT License

---

