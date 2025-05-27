# Health Risk Predictor – Diabetes & Heart Disease

This is a machine learning web app built with **Streamlit** that predicst the risk of **diabetes** and **Heart Disease** based on user input. It also explains each prediction using **SHAP values** that can be visualized with a bar plot.

## Features

- Predicts binary risk: Positice or Negative
- Input forms for:
  - **Diabetes**
  - **Heart Disease**
- Shows prediction probability
- Visualizes top contributing featrues using SHAP
- Automatically adapts UI and backend based on selected condition

## Tech Stack

- Frontend: Streamlit
- Backend: scikit-learn, pandas, numpy
- Modeling:
  - Diabetes: Random Forest + Scaler
  - Heart Disease: Pipeline (preprocessing + Random Forest)
- Explainability: SHAP (with static bar plots)

## Demo

[Live App Link](https://health-risk.streamlit.app/)

## Credits

- Dataset sources:
  - [Diabetes](https://www.kaggle.com/datasets/shahnawaj9/diabetes-database/data)
  - [Heart Disease](https://www.kaggle.com/datasets/denysskyrda/common-heart-disease-data-4-hospitals/data)
