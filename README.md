# 🛍️ E-commerce Visitor Purchase Prediction 

### **Project Overview**
This project is a Machine Learning-powered web application designed to predict whether an online shopper will generate revenue based on their session behavior. By analyzing metrics such as bounce rates, page values, and session duration, the model identifies high-intent visitors in real-time.

---
### ** Live Link :** https://ecommerce-prediction.streamlit.app/#high-conversion-intent
### **🛠️ Technical Stack**
* **Language:** Python 3.12 (64-bit)
* **Machine Learning:** K-Nearest Neighbors (KNN)
* **Optimization:** Principal Component Analysis (PCA) 
* **Data Balancing:** SMOTE (imbalanced-learn)
* **UI Framework:** Streamlit

---

### **🚀 Key Features**
* **Real-time Prediction:** Interactive dashboard for instant behavioral analysis.
* **Dimensionality Reduction:** Uses PCA to maintain model efficiency while handling 17+ features.
* **Professional UI:** Clean, sidebar-based navigation with visual probability metrics.

---

### **📂 Project Structure**
* `app.py`: The Streamlit web application interface.
* `train_model.py`: Script for data cleaning, SMOTE application, and model training.
* `online_shoppers_intention.csv`: The raw dataset from UCI Repository.
* `knn_model.pkl`: The saved trained KNN model.
* `pca_transformer.pkl`: The saved PCA configuration.
* `scaler.pkl`: The saved Standard Scaler for feature normalization.
* `requirements.txt`: Configuration for cloud deployment.

---

### **💻 How to Run Locally**

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sahilsingh682/ecommerce-prediction-ml.git](https://github.com/sahilsingh682/ecommerce-prediction-ml.git)
