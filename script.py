import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import matplotlib.pyplot as plt
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Page config for title, icon, and wide layout
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Function to add background image using base64 encoding
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use the function to set the background image
add_bg_from_local("background.jpg")

# Black Glassmorphism CSS styling
st.markdown(
    """
    <style>
    .css-1aumxhk, .css-1v3fvcr, .stButton, .stAlert {
        background: rgba(0, 0, 0, 0.85) !important; 
        border-radius: 15px;
        padding: 10px;
        color: white;  
    }
    .stSidebar {
        backdrop-filter: blur(10px); 
        background: rgba(0, 0, 0, 0.75); 
        border-radius: 15px;
        padding: 10px;
    }
    .stTextInput, .stNumberInput, .stRadio, .stSelectbox {
        backdrop-filter: blur(10px);
        background: rgba(0, 0, 0, 0.544) !important; 
        border-radius: 10px;
        padding: 10px;
        color: white;  
    }
    .main-title h1, .knn-subheader h2 {
        color: white !important;  
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
    }
    .stButton button {
        background-color: #0a0b0b;  
        color: white;  
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
    }
    .description-card {
        background: rgba(0, 0, 0, 0.75); 
        border-radius: 15px;
        padding: 20px;
        color: white;  
        margin: 20px 0; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
heart_data = pd.read_csv("heart.csv")

# Prepare features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Main title for the app
st.title("Heart Disease Prediction System")

# Description content
description = """
Welcome to the Heart Disease Prediction System, an innovative tool designed to assist individuals in assessing their risk of heart disease based on key health indicators. Our application leverages machine learning algorithms to provide accurate predictions and insights, empowering users to take proactive steps towards better heart health.

### Key Features:
- **User-Friendly Interface:** Easily input your health data in a streamlined sidebar form.
- **Predictive Modeling:** Choose between two robust algorithms—K-Nearest Neighbors (KNN) and Decision Tree—to evaluate your risk.
- **Detailed Insights:** View your input data in graphical form, along with model predictions and accuracy rates.
- **Feature Importance Analysis:** Understand which health factors contribute most significantly to your risk of heart disease through comprehensive feature importance results.
- **Personalized Feedback:** Receive tailored messages based on your prediction results to encourage a healthy lifestyle.

### How It Works:
1. **Input Your Data:** Enter your age, gender, blood pressure, cholesterol levels, and other relevant health metrics.
2. **Select an Algorithm:** Choose your preferred predictive model to analyze the data.
3. **Receive Predictions:** Get immediate feedback on your heart disease risk along with accuracy rates.
4. **Explore Feature Importance:** Discover which health indicators have the most influence on your prediction.

With a focus on clarity and usability, our Heart Disease Prediction System aims to bridge the gap between complex medical data and user-friendly technology. By making informed health decisions, you can work towards a healthier future.
"""

# Display the description in a card format
st.markdown(f"""
<div class="description-card">
    {description}
</div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.title("Enter Input Values")
form = st.sidebar.form(key='my_form')

# Input fields for prediction with ranges in brackets
age = form.number_input(label="Age (0-120)", min_value=0, max_value=120)
gender = form.radio("Gender", ["Male", "Female"])
cp = form.number_input(label="Chest Pain (0-3)", min_value=0, max_value=3)
bp = form.number_input(label="Resting Blood Pressure (0-200)", min_value=0, max_value=200)
chol = form.number_input(label="Serum Cholesterol (0-600)", min_value=0, max_value=600)
fbs = form.number_input(label="Fasting Blood Sugar (0-1)", min_value=0, max_value=1)
restecg = form.number_input(label="Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2)
thalch = form.number_input(label="Max Heart Rate (0-220)", min_value=0, max_value=220)
exang = form.number_input(label="Exercise-induced Angina (0-1)", min_value=0, max_value=1)
oldpeak = form.number_input(label="ST Depression (0.0-6.0)", min_value=0.0, max_value=6.0, format="%.2f")
slope = form.number_input(label="Slope (0-2)", min_value=0, max_value=2)
ca = form.number_input(label="Number of Major Vessels (0-4)", min_value=0, max_value=4)
thal = form.number_input(label="Thalassemia (0-3)", min_value=0, max_value=3)

# Algorithm selection
select_algorithm = form.selectbox("Select Algorithm", ["KNN", "Decision Tree"])
submit_button = form.form_submit_button(label='Submit')

# Convert gender to numeric
gender = 1 if gender == "Male" else 0

# If the form is submitted
if submit_button:
    # Reshape input data
    input_data = (int(age), int(gender), int(cp), int(bp), int(chol), int(fbs), int(restecg), int(thalch), int(exang), 
                  float(oldpeak), int(slope), int(ca), int(thal))
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Loading spinner for processing
    with st.spinner("Processing..."):
        # Select and train model
        if select_algorithm == "KNN":
            model = KNeighborsClassifier()
            model.fit(X_train, Y_train)
            prediction = model.predict(input_data_as_numpy_array)
            accuracy = accuracy_score(Y_test, model.predict(X_test))

            # Display input data graph for KNN
            st.subheader("Input Data Graph for KNN")
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(input_data)), input_data, tick_label=X.columns, color='black')
            plt.xticks(rotation=45)
            plt.ylabel("Input Value")
            plt.title("Input Data for KNN Prediction")
            st.pyplot(plt)

        elif select_algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
            model.fit(X_train, Y_train)
            prediction = model.predict(input_data_as_numpy_array)
            accuracy = accuracy_score(Y_test, model.predict(X_test))

        # Confusion Matrix
        cm = confusion_matrix(Y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        # Display the confusion matrix
        st.subheader("Confusion Matrix")
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot(plt)

    # Display the prediction and accuracy
    st.markdown("---")
    if prediction[0] == 1:
        st.subheader("Result: Positive")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        # Feature importance using Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, Y_train)
        feature_importances = rf_model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]

        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': X.columns[sorted_indices],
            'Importance Score': feature_importances[sorted_indices]
        })

        # Show feature importance
        st.subheader("Feature Importance")
        st.dataframe(importance_df)

    else:
        st.subheader("Result: Negative")
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Personalized message
    st.markdown("### Personalized Message")
    st.success("BE HAPPY, BE HEALTHY! REGARDS: PAYAL, PRASTUTI, MAYURI, ABHIBHOO")
