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
    st.title("🔍 Predictive Model for Heart Disease Mortality")
    st.write("""
    Welcome to the Heart Disease Prediction Dashboard!
    
    This dashboard provides an interactive analysis of heart disease patient data, aiming to predict mortality risk using advanced machine learning models.
    """)
    
    # Author and Model Information
    st.markdown("""
    ### Model Selection
    - The final model chosen for this project is **HistGradientBoostingClassifier**, selected for its high precision and balanced performance.
    
    ### About the Author
    - Created by: [Chris Chalfoun](https://www.linkedin.com/in/chris-chalfoun/)
    - LinkedIn: [Connect with me](https://www.linkedin.com/in/chris-chalfoun/)
    - Project Link: [Github Repo](https://github.com/Pointfit/healthcare-analytics-project)

    """)

# Data Exploration Section
elif options == "Data Exploration":
    st.header("Data Exploration")
    
    # Display a sample of the data
    st.write("**Sample of the Data**")
    st.write("""
    Here, we present a small sample of the dataset used in this project. Each row represents a patient, and each column provides information about the patient's characteristics or health metrics. This is helpful to get a quick understanding of the structure of the data we're working with.
    """)
    st.dataframe(heart_data.head())
    
    # Display data types and check for missing values
    st.write("**Data Types and Missing Values Check**")
    st.write("""
    Each column in the dataset has a specific data type:
    - **Numeric** columns (e.g., Age, Cholesterol) contain continuous numbers.
    - **Categorical** columns (e.g., ChestPainType, RestingECG) represent discrete categories.
    
    This check ensures that each feature has the correct data type for analysis. Additionally, we verify if there are any missing values in the dataset, which is important because missing data can affect model performance and accuracy.
    """)
    st.write(heart_data.dtypes)
    st.write(f"No missing values: {heart_data.isnull().sum().sum() == 0}")
    
    # Enhanced Data Visualizations

    # Scatter Plot - Age vs. Cholesterol Levels
    st.subheader("Age vs. Cholesterol Levels")
    st.write("""
    This scatter plot shows the relationship between **Age** and **Cholesterol Levels** for each patient. 
    - **Color-coded by Outcome**: Points are color-coded by the presence or absence of heart disease, making it easier to see if there’s a visual pattern.
    - This chart helps us visually inspect if older patients or those with higher cholesterol are more likely to have heart disease.
    """)
    fig = px.scatter(heart_data, x="Age", y="Cholesterol", color="HeartDisease", 
                     title="Age vs. Cholesterol Levels by Outcome")
    st.plotly_chart(fig)

    # Histogram - Distribution of Age
    st.subheader("Age Distribution")
    st.write("""
    This histogram shows the **distribution of ages** in the dataset.
    - **Peaks and Spreads**: The bars represent the number of patients within specific age ranges. This visualization helps us understand the age composition of our dataset.
    - Such distributions are useful to see if the dataset is balanced across ages or if certain age groups are overrepresented.
    """)
    fig = px.histogram(heart_data, x="Age", nbins=30, title="Distribution of Age")
    st.plotly_chart(fig)

    # Box Plot - Cholesterol Levels by Outcome
    st.subheader("Cholesterol Levels by Outcome")
    st.write("""
    This box plot compares **cholesterol levels** for patients with and without heart disease.
    - **Box and Whiskers**: Each box represents the spread of cholesterol levels (25th to 75th percentile), with the line in the middle showing the median. The "whiskers" extend to show the range of most data points.
    - **Purpose**: This visualization highlights any differences in cholesterol levels between the two groups, helping us see if higher cholesterol might be linked to heart disease.
    """)
    fig = px.box(heart_data, x="HeartDisease", y="Cholesterol", title="Cholesterol Levels by Outcome")
    st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("""
    This heatmap displays **correlations between numerical features** in the dataset.
    - **Correlation** measures how closely related two variables are (ranges from -1 to +1). Positive values indicate a direct relationship, while negative values suggest an inverse relationship.
    - **Purpose**: Identifying strong correlations can help us understand which features are closely linked. For instance, if "Age" and "Heart Disease" show a strong correlation, age could be an important factor in our model.
    """)
    numeric_data = heart_data.select_dtypes(include=[np.number])  # Select only numeric columns
    corr = numeric_data.corr()  # Calculate correlation on numeric data only
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Model Comparison Section
elif options == "Model Comparison":
    st.header("Model Comparison")
    st.write("""
    In this section, we compare the performance of different machine learning models used to predict heart disease. 
    Each model was evaluated using key performance metrics, allowing us to see how well each model performs and select the best one.
    """)

    # Model performance metrics from your evaluations
    st.write("**Model Performance Metrics**")
    st.write("""
    The table below shows the performance of four different models on the test set, using four key metrics:
    - **Precision**: Measures how many of the positive predictions made by the model are actually correct. A high precision means fewer false positives (patients incorrectly labeled as "at risk").
    - **Recall**: Measures how well the model identifies actual positives. A high recall means the model is good at detecting most of the actual "at risk" patients.
    - **F1-Score**: This is the balance between precision and recall, useful when both are important. A higher F1-score indicates a good trade-off between precision and recall.
    - **Accuracy**: The percentage of all correct predictions made by the model. While helpful, accuracy alone can be misleading if there’s an imbalance in the dataset (e.g., more healthy patients than at-risk ones).
    
    The best-performing model, based on a balance of all metrics, will likely give the most reliable predictions in a real-world setting.
    """)
    
    model_performance = pd.DataFrame({
        "Model": ["Gradient Boosting (Initial)", "Gradient Boosting (Tuned)", "HistGradientBoostingClassifier (Final)", "CatBoost"],
        "Precision (Test)": [0.90, 0.90, 0.92, 0.89],
        "Recall (Test)": [0.85, 0.86, 0.87, 0.86],
        "F1-Score (Test)": [0.88, 0.88, 0.89, 0.86],
        "Accuracy (Test)": [0.86, 0.86, 0.88, 0.86]
    })
    st.write(model_performance)

    # Explanation of the Table
    st.write("""
    Each row in the table represents a different model:
    - **Gradient Boosting (Initial)**: This is the initial version of the Gradient Boosting model.
    - **Gradient Boosting (Tuned)**: This version has been fine-tuned to improve its performance.
    - **HistGradientBoostingClassifier (Final)**: This is the final selected model due to its best balance across metrics.
    - **CatBoost**: A separate model type known for handling categorical features well.
    
    The numbers indicate each model's performance across the four metrics, making it easy to see which model performed best.
    """)

    # Bar plot for model performance
    st.subheader("Model Performance Comparison")
    st.write("""
    The bar chart below visualizes the comparison between models for Precision, Recall, and F1-Score. 
    - **Grouped Bars**: Each group of bars represents a different model, allowing us to visually compare their performance on each metric.
    - This visual representation makes it easier to identify which model excels in specific areas, such as having higher precision or recall.
    """)
    
    fig = px.bar(model_performance, x="Model", y=["Precision (Test)", "Recall (Test)", "F1-Score (Test)"], 
                 barmode='group', title="Model Performance Comparison by Metric")
    fig.update_layout(xaxis_title="Model", yaxis_title="Score", legend_title="Metric")
    st.plotly_chart(fig)

# Final Model Results Section - HistGradientBoostingClassifier
elif options == "Final Model Performance":
    st.header("Final Model: HistGradientBoostingClassifier Performance")
    st.write("""
    This section provides an in-depth look at the performance of our final model, the **HistGradientBoostingClassifier**. 
    This model was selected due to its strong performance in balancing precision and recall, making it a reliable choice 
    for identifying heart disease risk.
    
    Here, we evaluate the model using key metrics such as **Precision**, **Recall**, **F1-Score**, and **Accuracy** to 
    gauge its effectiveness in correctly identifying patients who are at risk of heart disease.
    """)

    # Example true labels and predictions for the final model
    y_test = [0] * 77 + [1] * 107  # True labels based on test set
    test_predictions = [0] * 69 + [1] * 8 + [0] * 14 + [1] * 93  # Model predictions based on confusion matrix

    # Calculate and display metrics for the final model
    st.subheader("Final Model Performance Metrics")
    st.write("""
    Below are the final performance metrics for the HistGradientBoostingClassifier on the test dataset:
    - **Precision**: 92% - The model correctly identifies 92% of "at risk" cases out of all the positive predictions it made.
    - **Recall**: 87% - The model successfully detects 87% of the actual "at risk" cases, ensuring it captures most people who truly need attention.
    - **F1-Score**: 89% - Balances both precision and recall, providing a single metric to assess model reliability.
    - **Accuracy**: 88% - Overall, the model correctly classified 88% of cases, but remember, accuracy alone doesn’t capture the model’s performance well if classes are imbalanced.
    """)
    
    st.write(f"**Precision (Test):** 0.92")
    st.write(f"**Recall (Test):** 0.87")
    st.write(f"**F1-Score (Test):** 0.89")
    st.write(f"**Accuracy (Test):** 0.88")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write("""
    The confusion matrix below illustrates how well the model distinguishes between "Healthy" and "At Risk" cases:
    - **True Positives (Top-right cell)**: Patients correctly identified as "At Risk".
    - **False Positives (Bottom-right cell)**: Patients incorrectly labeled as "At Risk" but are actually healthy.
    - **True Negatives (Top-left cell)**: Patients correctly identified as healthy.
    - **False Negatives (Bottom-left cell)**: Patients who are actually "At Risk" but were missed by the model.
    
    This matrix gives a more detailed breakdown of where the model performs well and where it may make errors.
    """)
    
    cm = confusion_matrix(y_test, test_predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    st.write("""
    The ROC (Receiver Operating Characteristic) Curve below shows the trade-off between the **True Positive Rate** and the 
    **False Positive Rate** as the decision threshold for the model changes. 
    - The **AUC (Area Under Curve)** score summarizes the model's ability to distinguish between "Healthy" and "At Risk" cases.
    - A higher AUC score (closer to 1) indicates a better model. In our case, the AUC score is quite high, signifying the model's strong performance.
    
    This visualization helps us understand how well the model separates the two classes (Healthy vs. At Risk).
    """)
    
    fpr, tpr, _ = roc_curve(y_test, test_predictions)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
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

    if st.button("Predict", key="predict_button"):
        # Get probability prediction
        probability = model.predict_proba(input_data)[0][1]  # Probability for "At Risk" class
        st.write("Prediction Probability:", probability)
        
        # Interpret based on probability threshold
        threshold = 0.5
        if probability >= threshold:
            result_text = "At Risk (Higher likelihood of heart disease)"
        else:
            result_text = "Healthy (Low likelihood of heart disease)"
        
        st.write("Prediction:", result_text)


# Conclusion Section
else:
    st.header("📈 Conclusion and Recommendations")
    st.write("""
    The **HistGradientBoostingClassifier** model demonstrated high precision and balanced recall, making it a reliable tool for predicting mortality risk in heart disease patients.

    ### Key Insights:
    - The model provides robust performance, making it well-suited for mortality prediction tasks in clinical settings.
    - Achieving balanced recall and precision highlights the model's effectiveness in identifying "At Risk" patients without over-predicting.

    ### Future Directions:
    - **Expand the Dataset**: Incorporate more diverse patient data to improve model generalizability.
    - **Feature Engineering**: Explore additional features that may enhance model performance and insights.
    - **Clinical Deployment**: With proper validation, this model could be integrated into healthcare systems to assist clinicians in identifying at-risk patients early.

    """)
    st.markdown("#### Thank you for exploring this project! For more, connect with me on [LinkedIn](https://www.linkedin.com/in/chris-chalfoun/).")

