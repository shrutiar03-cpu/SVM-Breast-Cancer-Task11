# Task 11: SVM – Breast Cancer Classification
## 📌 Objective
To build and evaluate an SVM model for classifying breast cancer as malignant or benign using Scikit-Learn.

## 📂 Dataset
- Used: `load_breast_cancer()` dataset from sklearn
- 569 samples, 30 features
- Labels: 
  - 0 → Malignant  
  - 1 → Benign  

## ⚙️ Steps Followed
1. Loaded and inspected the dataset.
2. Applied StandardScaler for feature normalization.
3. Split data into train and test sets (80–20 split).
4. Trained baseline SVM with **linear kernel**.
5. Trained SVM with **RBF kernel** and compared accuracy.
6. Used **GridSearchCV** to tune C and gamma.
7. Evaluated best model using:
   - Confusion Matrix  
   - Classification Report  
   - ROC Curve & AUC Score  
8. Saved final model pipeline (`scaler.pkl` + `svm_model.pkl`).

## 🧠 Results
- Linear SVM Accuracy: ~97%
- Tuned RBF SVM Accuracy: ~98% (varies slightly)
- AUC Score: ~0.99

## 📁 Files in this Repository
- `SVM_Breast_Cancer_Task11.ipynb`
- `scaler.pkl`
- `svm_model.pkl`
- Screenshots (if added)

## 🛠 Tools Used
- Python  
- Scikit-Learn  
- Matplotlib
