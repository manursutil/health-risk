import joblib

def load_model(dataset):
    if dataset == "diabetes":
        model = joblib.load("models/diabetes_rf.pkl")
        scaler = joblib.load("models/diabetes_scaler.pkl")
        features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    elif dataset == "heart":
        model = joblib.load("models/heart_rf.pkl")
        scaler = joblib.load("models/heart_scaler.pkl")
        features = ["age", "trestbps", "chol", "thalach", "oldpeak", "cp_2", "cp_3", "cp_4", "exang", "slope_2", "slope_3", "thal_6", "thal_7", "ca"]
    else:
        raise ValueError("Unsupported dataset. Please choose 'diabetes' or 'heart'.")
    
    return model, scaler, features

