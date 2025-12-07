import pickle
import pandas as pd
import numpy as np


with open("models\logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models\Feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


def encode_input(user_dict):
    x = np.zeros(len(feature_columns))
    
    # Numeric columns
    x[feature_columns.index('tenure')] = user_dict['tenure']
    x[feature_columns.index('MonthlyCharges')] = user_dict['MonthlyCharges']
    x[feature_columns.index('TotalCharges')] = user_dict['TotalCharges']
    
    # InternetService
    if user_dict['InternetService'] == 'Fiber optic':
        x[feature_columns.index('InternetService_Fiber optic')] = 1
    elif user_dict['InternetService'] == 'No':
        x[feature_columns.index('InternetService_No')] = 1
    # DSL is baseline
    
    # OnlineSecurity
    if user_dict['OnlineSecurity'] == 'Yes':
        x[feature_columns.index('OnlineSecurity_Yes')] = 1
    elif user_dict['OnlineSecurity'] == 'No internet service':
        x[feature_columns.index('OnlineSecurity_No internet service')] = 1
    
    # OnlineBackup
    if user_dict['OnlineBackup'] == 'No internet service':
        x[feature_columns.index('OnlineBackup_No internet service')] = 1
    
    # DeviceProtection
    if user_dict['DeviceProtection'] == 'No internet service':
        x[feature_columns.index('DeviceProtection_No internet service')] = 1
    
    # TechSupport
    if user_dict['TechSupport'] == 'No internet service':
        x[feature_columns.index('TechSupport_No internet service')] = 1
    
    # StreamingTV
    if user_dict['StreamingTV'] == 'No internet service':
        x[feature_columns.index('StreamingTV_No internet service')] = 1
    
    # StreamingMovies
    if user_dict['StreamingMovies'] == 'No internet service':
        x[feature_columns.index('StreamingMovies_No internet service')] = 1
    
    # Contract
    if user_dict['Contract'] == 'One year':
        x[feature_columns.index('Contract_One year')] = 1
    elif user_dict['Contract'] == 'Two year':
        x[feature_columns.index('Contract_Two year')] = 1
    # Month-to-month is baseline
    
    # PaymentMethod
    if user_dict['PaymentMethod'] == 'Electronic check':
        x[feature_columns.index('PaymentMethod_Electronic check')] = 1
    
    
    return pd.DataFrame([x], columns=feature_columns)


def predict_churn(user_dict, threshold=0.4):
    X_input = encode_input(user_dict)
    y_prob = model.predict_proba(X_input)[:,1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "Prediction": "Churn" if y_pred[0] == 1 else "No Churn",
        "Probability": float(y_prob[0])
    }


if __name__ == "__main__":
    user_input = {
        'tenure': 12,
        'MonthlyCharges': 75.5,
        'TotalCharges': 910.5,
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No internet service',
        'TechSupport': 'No internet service',
        'StreamingTV': 'No internet service',
        'StreamingMovies': 'No internet service',
        'Contract': 'One year',
        'PaymentMethod': 'Electronic check'
    }
    
    result = predict_churn(user_input)
    print(f"Prediction: {result['Prediction']}, Probability: {result['Probability']:.2f}")
