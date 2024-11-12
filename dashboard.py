import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc
import numpy as np

# Load your data
heart_data = pd.read_csv('heart.csv')  # Ensure 'heart.csv' is in the same directory or update the path

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["Project Overview", "Data Exploration", "Model Comparison", "Final Model Performance", "Conclusion"])

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
    st.write("**Sample of the Data**")
    st.dataframe(heart_data.head())
    st.write("**Data Types and Missing Values Check**")
    st.write(heart_data.dtypes)
    st.write(f"No missing values: {heart_data.isnull().sum().sum() == 0}")

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

    # Replace the placeholders with the actual constructed lists for accurate confusion matrix and ROC
    y_test = [0] * 77 + [1] * 107  # True labels based on test set
    test_predictions = [0] * 69 + [1] * 8 + [0] * 14 + [1] * 93  # Model predictions based on confusion matrix

    # Precision, Recall, and F1-Score for HistGradientBoostingClassifier
    st.write(f"**Precision (Test):** 0.92")
    st.write(f"**Recall (Test):** 0.87")
    st.write(f"**F1-Score (Test):** 0.89")
    st.write(f"**Accuracy (Test):** 0.88")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, test_predictions)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center')
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, test_predictions)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig)

# Conclusion Section
else:
    st.header("Conclusion and Recommendations")
    st.write("""
    The HistGradientBoostingClassifier achieved the highest precision and balanced recall, making it a robust choice for mortality prediction.
    Future improvements could include expanding the dataset, feature engineering, and deploying the model in a clinical setting.
    """)
