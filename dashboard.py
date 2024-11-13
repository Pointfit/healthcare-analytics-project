import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import numpy as np
import seaborn as sns
import joblib

# Load your data
heart_data = pd.read_csv('heart.csv')  # Ensure 'heart.csv' is in the same directory or update the path

# Load the saved model
model = joblib.load('final_model.pkl')

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["Project Overview", "Data Exploration", "Model Comparison", "Final Model Performance", "Prediction Tool", "Conclusion"])

# Project Overview Section
if options == "Project Overview":
    st.title("Predictive Model for Heart Disease Mortality")
    st.write("""
    This dashboard provides an analysis of heart disease patient data, with a focus on predicting mortality using machine learning models.
    The final model, HistGradientBoostingClassifier, was selected for its precision and balanced performance.
    """)

# Data Exploration Section
elif options == "Data Exploration":
    st.header("Data Exploration")
    
    # Display a sample of the data
    st.write("**Sample of the Data**")
    st.dataframe(heart_data.head())
    
    # Display data types and check for missing values
    st.write("**Data Types and Missing Values Check**")
    st.write(heart_data.dtypes)
    st.write(f"No missing values: {heart_data.isnull().sum().sum() == 0}")
    
    # Enhanced Data Visualizations
    
    # Scatter Plot - Age vs. Cholesterol Levels
    st.subheader("Age vs. Cholesterol Levels")
    fig = px.scatter(heart_data, x="Age", y="Cholesterol", color="HeartDisease", 
                     title="Age vs. Cholesterol Levels by Outcome")
    st.plotly_chart(fig)

    # Histogram - Distribution of Age
    st.subheader("Age Distribution")
    fig = px.histogram(heart_data, x="Age", nbins=30, title="Distribution of Age")
    st.plotly_chart(fig)

    # Box Plot - Cholesterol Levels by Outcome
    st.subheader("Cholesterol Levels by Outcome")
    fig = px.box(heart_data, x="HeartDisease", y="Cholesterol", title="Cholesterol Levels by Outcome")
    st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_data = heart_data.select_dtypes(include=[np.number])  # Select only numeric columns
    corr = numeric_data.corr()  # Calculate correlation on numeric data only
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Model Comparison Section
elif options == "Model Comparison":
    st.header("Model Comparison")
    st.write("Comparison of model performance metrics on the test set.")

    # Model performance metrics from your evaluations
    model_performance = pd.DataFrame({
        "Model": ["Gradient Boosting (Initial)", "Gradient Boosting (Tuned)", "HistGradientBoostingClassifier (Final)", "CatBoost"],
        "Precision (Test)": [0.90, 0.90, 0.92, 0.89],
        "Recall (Test)": [0.85, 0.86, 0.87, 0.86],
        "F1-Score (Test)": [0.88, 0.88, 0.89, 0.86],
        "Accuracy (Test)": [0.86, 0.86, 0.88, 0.86]
    })

    st.write(model_performance)

    # Bar plot for model performance
    fig = px.bar(model_performance, x="Model", y=["Precision (Test)", "Recall (Test)", "F1-Score (Test)"], barmode='group')
    st.plotly_chart(fig)

# Final Model Results Section - HistGradientBoostingClassifier
elif options == "Final Model Performance":
    st.header("Final Model: HistGradientBoostingClassifier Performance")

    # Example true labels and predictions for the final model
    y_test = [0] * 77 + [1] * 107  # True labels based on test set
    test_predictions = [0] * 69 + [1] * 8 + [0] * 14 + [1] * 93  # Model predictions based on confusion matrix

    # Calculate and display metrics for the final model
    st.write(f"**Precision (Test):** 0.92")
    st.write(f"**Recall (Test):** 0.87")
    st.write(f"**F1-Score (Test):** 0.89")
    st.write(f"**Accuracy (Test):** 0.88")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, test_predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, test_predictions)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Prediction Tool Section
elif options == "Prediction Tool":
    st.header("Prediction Tool")
    st.write("Enter the necessary inputs and use the trained model to make predictions.")

    # Input fields for each feature with context to aid user understanding
    age = st.number_input("Enter Age:", min_value=0, max_value=120, help="The age of the patient, in years. (e.g., 45)")
    sex = st.selectbox("Select Sex:", options=[1, 0], help="The biological sex of the patient. 1 represents Male, and 0 represents Female.")
    chest_pain_type = st.selectbox("Chest Pain Type:", options=[0, 1, 2, 3], 
                                   help="""Type of chest pain experienced:
                                   - 0: Typical angina - chest pain related to decreased blood supply to the heart
                                   - 1: Atypical angina - chest pain not related to the heart
                                   - 2: Non-anginal pain - not heart-related chest pain
                                   - 3: Asymptomatic - no chest pain""")
    resting_bp = st.number_input("Enter Resting Blood Pressure:", min_value=80, max_value=200, 
                                 help="Patient's resting blood pressure, measured in mm Hg. Normal range is 120/80 mm Hg.")
    cholesterol = st.number_input("Enter Cholesterol Level:", min_value=50, max_value=600, 
                                  help="Patient's serum cholesterol level in mg/dL. Normal is less than 200 mg/dL.")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL:", options=[1, 0], 
                              help="Indicates if fasting blood sugar is greater than 120 mg/dL. 1 means Yes, 0 means No.")
    resting_ecg = st.selectbox("Resting ECG:", options=[0, 1, 2], 
                               help="""Results of the patient's resting electrocardiogram (ECG):
                               - 0: Normal - No abnormalities
                               - 1: ST-T wave abnormality - possible indication of heart strain or old heart attack
                               - 2: Left ventricular hypertrophy - thickening of the heart's left ventricle""")
    max_hr = st.number_input("Enter Maximum Heart Rate:", min_value=60, max_value=220, 
                             help="Maximum heart rate achieved during exercise. Normal ranges vary but often between 140-190 bpm depending on age.")
    exercise_angina = st.selectbox("Exercise Induced Angina:", options=[1, 0], 
                                   help="Indicates if exercise caused chest pain (angina). 1 means Yes, 0 means No.")
    oldpeak = st.number_input("Enter Oldpeak:", min_value=0.0, max_value=10.0, step=0.1, 
                              help="ST depression induced by exercise relative to rest, measured in mm. Indicates possible heart ischemia.")
    st_slope = st.selectbox("Select ST Slope:", options=[0, 1, 2], 
                            help="""The slope of the peak exercise ST segment:
                            - 0: Upsloping - generally associated with healthier outcomes
                            - 1: Flat - may indicate some risk of heart disease
                            - 2: Downsloping - often associated with higher risk of heart disease""")

    # Prepare input data in the correct order
    input_data = [[
        age, sex, chest_pain_type, resting_bp, cholesterol,
        fasting_bs, resting_ecg, max_hr, exercise_angina,
        oldpeak, st_slope
    ]]

    # Prediction button and model prediction code
    if st.button("Predict"):
        # Use the loaded model to make a prediction
        prediction = model.predict(input_data)[0]  # Get the prediction

        # Interpret the prediction result
        if prediction == 0:
            result_text = "Healthy (Low likelihood of heart disease)"
        else:
            result_text = "At Risk (Higher likelihood of heart disease)"
        
        st.write("Prediction:", result_text)

# Conclusion Section
else:
    st.header("Conclusion and Recommendations")
    st.write("""
    The HistGradientBoostingClassifier achieved the highest precision and balanced recall, making it a robust choice for mortality prediction.
    Future improvements could include expanding the dataset, feature engineering, and deploying the model in a clinical setting.
    """)
