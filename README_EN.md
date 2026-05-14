# HeartGuard ❤️🛡️

HeartGuard is a real-time **Risk Analysis and Dashboard** application that analyzes clinical ECG (Electrocardiography) signals and vital signs using deep learning and machine learning models. It is specifically designed to detect Myocardial Infarction (Heart Attack - STEMI/NSTEMI) and rhythm disorders (Arrhythmia).

## 📊 Databases Used (Training Data)

The project's AI models and live tests are based on globally recognized clinical medical datasets:

1. **PTB-XL Database:**
   - The heart of the deep learning model is based on this dataset. It contains 10-second 12-lead high-quality clinical ECG recordings from 21,837 patients.
   - In the project, a **1D CNN (Convolutional Neural Network)** model is trained in `train_ptbxl_model.py` using this data to distinguish between "Normal" and "MI (Myocardial Infarction)".
2. **MIT-BIH Arrhythmia Database:**
   - Used for real-time testing and simulations on the Dashboard (`main.py`). The system dynamically fetches recordings from the database using the `wfdb` library to live-analyze different types of arrhythmias ranging from normal sinus rhythm to ventricular fibrillation.
3. **Synthetic Data Generation:**
   - Artificial data generation algorithms simulating specific heart attack findings such as ST Segment Elevation (STEMI) for test scenarios.
4. **Vital Signs:**
   - There is a Random Forest model that evaluates and classifies the patient's Pulse, Systolic, and Diastolic Blood Pressure values to determine risk based on vital tables.

## ⚙️ Process: How Does the System Work?

1. **Model Training (Deep Learning):**
   - ECG signals are processed via `train_ptbxl_model.py` and `train_cnn_model.py`. Features are extracted from time-series ECG waves using 1D CNN layers.
   - The trained large-sized Keras (`.h5`) models are optimized and converted into **TensorFlow Lite (`.tflite`)** format to run completely offline with minimal latency on the desktop application.

2. **Real-Time Analysis (Dashboard):**
   - When `main.py` is executed, a Tkinter desktop interface opens.
   - The system receives incoming ECG signals simulating a live patient feed.
   - Signals are divided into specific "window" sizes (e.g., 750 samples) and passed through the `TFLite` model using the sliding window method.
   - The model produces an "Anomaly / Risk Score" between 0 and 1 for each section.

3. **Risk Assessment and Visualization:**
   - If the anomaly score exceeds the critical threshold (e.g., 0.6), the system marks the exact section on the ECG graph using Matplotlib with **arrows and risk scores** on the interface.
   - All ECG analysis results, along with the patient's vital signs, are combined and reflected on the screen as the **"Overall Heart Attack Risk"** (Low Risk, Medium Risk, Critical Danger, etc.).
   - All analysis stages are detailed sequentially in the "Analysis Log" tab in the interface.

## 🚀 Installation and Execution

### Requirements
- Python 3.8+
- To install the required libraries:
  ```bash
  pip install numpy pandas tensorflow wfdb scikit-learn matplotlib joblib
  ```

### Usage

To retrain and save the TFLite AI model from scratch (Optional):
```bash
python train_ptbxl_model.py
```

To run the ECG analysis and clinical interface (Dashboard):
```bash
python main.py
```
