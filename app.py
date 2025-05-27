from load_model import load_model
import streamlit as st
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(page_title="Health Risk Predictor", layout="centered")

st.title("Diabetes and Heart Disease Prediction")
st.markdown(
    """
    This application predicts risk of **diabetes** and **heart disease** based on user inputs.
    Select a condition below and enter your health data to get started.
    """
)

dataset = st.selectbox("Choose a condition:", ["Diabetes", "Heart Disease"])

key= "heart" if dataset == "Heart Disease" else "diabetes"
model, scaler, features = load_model(key)

st.subheader(f"Input your data for {dataset} prediction")

input_data = []

if dataset == "Diabetes":
    user_input = {
        "Pregnancies": st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0),
        "Glucose": st.number_input("Glucose Level", min_value=0, max_value=200, value=100),
        "BloodPressure": st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70),
        "SkinThickness": st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20),
        "Insulin": st.number_input("Insulin Level", min_value=0, max_value=500, value=80),
        "BMI": st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0),
        "DiabetesPedigreeFunction": st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5),
        "Age": st.number_input("Age", min_value=1, max_value=120, value=30)
    }

elif dataset == "Heart Disease":
    user_input = {
        "age": st.number_input("Age", min_value=1, max_value=120, value=30),
        "trestbps": st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120),
        "chol": st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200),
        "thalach": st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150),
        "oldpeak": st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0),
        "cp": st.selectbox("Chest Pain Type 2", [2, 3, 4]),
        "exang": st.selectbox("Exercise Induced Angina", [0, 1]),
        "slope": st.selectbox("Slope of ST Segment (Type 2)", [2, 3]),
        "thal": st.selectbox("Thalassemia Type", [6, 7]),
        "ca": st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    }

if st.button("🔍 Predict"):
    X_input_df = pd.DataFrame([user_input])
    
    if dataset == "Diabetes":
        X_scaled = scaler.transform(X_input_df[features])
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        pipeline = joblib.load("models/heart_rf_pipeline.pkl")
        prediction = pipeline.predict(X_input_df)[0]
        prob = pipeline.predict_proba(X_input_df)[0][1]
        
    st.subheader("🔎 Result")
    st.markdown(
        f"""
        **Prediction:** {'Positive' if prediction else 'Negative'}  
        **Probability:** {prob:.2%}
        """
    )

    try:
        if dataset == "Heart Disease":
            st.subheader("Top Features Impacting Prediction")

            # Load pipeline and training data
            pipeline = joblib.load("models/heart_rf_pipeline.pkl")
            X_train = pd.read_csv("data/heart_disease_cleaned.csv").drop(columns=["target", "source"])

            # Extract model and preprocessor
            model_only = pipeline.named_steps["classifier"]
            preprocessor = pipeline.named_steps["preprocessor"]

            # Transform the data
            X_train_transformed = preprocessor.transform(X_train).toarray()
            X_input_transformed = preprocessor.transform(X_input_df).toarray()

            # Get feature names after one-hot encoding
            cat_features = ["cp", "slope", "thal", "exang"]
            num_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
            ohe = preprocessor.named_transformers_["cat"]
            ohe_features = ohe.get_feature_names_out(cat_features)
            final_features = np.concatenate([num_features, ohe_features])

            # Build SHAP explainer
            explainer = shap.Explainer(model_only, X_train_transformed, feature_names=final_features)
            shap_values = explainer(X_input_transformed)

            # Extract SHAP values for the first sample
            shap_vals = shap_values.values[0, :, 1]
            
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
                base_value = base_value[1]
            elif isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
                
            prob = model_only.predict_proba(X_input_transformed)[0][1]


            top_n = 10
            sorted_idx = np.argsort(np.abs(shap_vals))[-top_n:]
            sorted_names = [final_features[i] for i in sorted_idx]
            sorted_vals = [shap_vals[i] for i in sorted_idx]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["green" if v < 0 else "red" for v in sorted_vals]

            ax.barh(sorted_names, sorted_vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"Top Features Impacting Prediction\nModel Output: {prob:.2f} | Base: {base_value:.2f}",
                        fontsize=13, weight='bold')
            ax.set_xlabel("SHAP Value (Impact on Prediction)", fontsize=11)
            ax.tick_params(axis='y', labelsize=10)

            fig.tight_layout()
            st.pyplot(fig)
            
        elif dataset == "Diabetes":
            st.subheader("Top Features Impacting Prediction")

            df_diabetes = pd.read_csv("data/diabetes_cleaned.csv")
            X_train = df_diabetes.drop(columns=["Outcome"])
            X_train_scaled = scaler.transform(X_train)
            X_input_scaled = scaler.transform(X_input_df[features])

            explainer = shap.Explainer(model, X_train_scaled, feature_names=features)
            shap_values = explainer(X_input_scaled)

            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
                base_value = base_value[1]
            elif isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]

            shap_vals = shap_values.values[0, :, 1] if shap_values.values.ndim == 3 else shap_values[0].values
            prob = model.predict_proba(X_input_scaled)[0][1]

            top_n = 10
            sorted_idx = np.argsort(np.abs(shap_vals))[-top_n:]
            sorted_names = [features[i] for i in sorted_idx]
            sorted_vals = [shap_vals[i] for i in sorted_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["green" if v < 0 else "red" for v in sorted_vals]
            ax.barh(sorted_names, sorted_vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"Top Features Impacting Prediction\nModel Output: {prob:.2f} | Base: {base_value:.2f}",
                         fontsize=13, weight='bold')
            ax.set_xlabel("SHAP Value (Impact on Prediction)", fontsize=11)
            ax.tick_params(axis='y', labelsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.info("SHAP explanation not available.")
        st.text(str(e))