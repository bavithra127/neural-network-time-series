# 🧠 Multivariate Time-Series Forecasting using Seq2Seq LSTM with Integrated Gradients

This project implements an **end-to-end deep learning pipeline** for multivariate time-series forecasting using a **Sequence-to-Sequence (Seq2Seq) LSTM model**. It also includes **model interpretability** using **Integrated Gradients**, providing insights into how each feature contributes to predictions.

---

## 📌 Features

### ✅ 1. Synthetic Multivariate Dataset
Time-series data with **trend**, **seasonality**, and **noise** across three features and one target variable.

### ✅ 2. Sliding Window Sequence Creation
- Input sequence length: **30 timesteps**
- Prediction horizon: **5 future timesteps**

### ✅ 3. Seq2Seq LSTM Model  
Includes:
- LSTM Encoder  
- RepeatVector  
- LSTM Decoder  
- TimeDistributed Dense layer  

### ✅ 4. Model Interpretation
Uses **Integrated Gradients** to compute feature importance.

### ✅ 5. Evaluation Metrics
- RMSE  
- MAPE  

### ✅ 6. Visualizations
- Loss curves  
- Feature importance chart  

---

## 🚀 How to Run the Project

### Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Run the Script
```bash
python neural_network_py.py
```

---

## 🧩 Key Functions

- `load_and_preprocess_data()`  
- `create_sequences()`  
- `create_seq2seq_model()`  
- `integrated_gradients()`  
- `main()`

---

## 📄 License
Free for academic and personal use.
