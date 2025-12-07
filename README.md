# 📞 Telco Customer Churn Prediction  
Machine Learning project to predict whether a telecom customer will churn based on their service usage, account details, and contract information.

## 📌 Overview
This project uses the **Telco Customer Churn dataset** to:
- Analyze customer behavior
- Identify patterns related to churn
- Build & evaluate a machine learning model to predict churn
- Optimize performance using threshold tuning
- Save the ML model for deployment

The project demonstrates end-to-end ML workflow:  
**Data Cleaning → EDA → Feature Engineering → Modeling → Threshold Optimization → Evaluation → Deployment.**

---

## 📊 Dataset
**Name:** Telco Customer Churn  
**Original Source:** Kaggle  
**File Used:** `dataset.csv` (original name : `WA_Fn-UseC_-Telco-Customer-Churn.csv`)

**Features include:**
- Customer demographics  
- Services subscribed (Internet, Phone, Security, Streaming)  
- Contract type  
- Billing details  
- Monthly & total charges  
- Target variable: `Churn`

---

## ⚙️ Project Workflow

### **1. Data Preprocessing**
- Removed irrelevant ID column (`customerID`)
- Converted `TotalCharges` to numeric
- Handled missing values
- Encoded categorical variables using one-hot encoding
- Split into train/test sets

---

## 📈 Exploratory Data Analysis (EDA)
Includes:
- Churn distribution  
- Visualizing numerical features  
- Correlation heatmap  
- Service-wise churn comparison

Key Insights:
- Month-to-month contracts have much higher churn  
- Fiber optic users churn more  
- Electronic check payment customers show higher churn

---

## 🤖 Model Building
**Model Used:** Logistic Regression  
Chosen because:
- Interpretable  
- Works well on binary classification  
- Fast and stable

---

## 🎯 Threshold Tuning
Instead of using the default 0.5 probability, multiple thresholds were tested (0.2 to 0.8).

**Selected Threshold:** **0.4**, because:
- Improved recall for churn customers  
- Better F1-score  
- Reduced false negatives

This step shows strong real-world ML understanding.

---

## 📉 Model Evaluation
Metrics included:
- Accuracy  
- Recall (priority for churn)  
- Precision  
- F1-score  
- Confusion matrix

---

## 💾 Model Saving
The trained model and feature order were saved using:
```python
pickle.dump(model, open('model.pkl', 'wb'))
