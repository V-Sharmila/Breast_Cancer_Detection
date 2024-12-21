import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define column names as per the dataset documentation
columns = [
    "ID", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Load the dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(data_url, header=None, names=columns)

# Drop the ID column as it's not a feature
data.drop(columns=["ID"], inplace=True)

# Map diagnosis column to binary values
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Select minimal features
minimal_features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]
X = data[minimal_features]
y = data['diagnosis']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# App UI
st.set_page_config(page_title="Breast Cancer Prediction", 
                   page_icon="🎗", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(to right, #FF80AB, #F50057);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;  /* Reduced margin to decrease space */
    }
    .instructions {
        text-align: center;
        font-size: 1.2em;
        color: white;
        margin-top: -10px;  /* Negative margin to further reduce space */
    }
    .stButton>button {
        background-color: #00C853;  /* Change to green */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #00E676;  /* Lighter green for hover effect */
    }
    .stNumberInput input {
        font-size: 1.2em;
        text-align: center;
    }
    .stAlert {
        background-color: #f8d7da;  /* Light red for error messages */
        border-color: #f5c6cb;
        color: #721c24;
    }
    .stSuccess {
        background-color: #d4edda;  /* Light green for success messages */
        border-color: #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Page title with white ribbon emoji next to it
st.markdown('<div class="title">Breast Cancer Prediction 🎗</div>', unsafe_allow_html=True)

# Instructions with reduced space
st.markdown("""
    <p class="instructions">Enter the following features and click on 'Predict' to get the results.</p>
""", unsafe_allow_html=True)

# Get minimal feature input from the user with number inputs
user_data = {}
for feature in minimal_features:
    user_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0, step=0.1)

# Prediction button with a nice hover effect
if st.button("Predict"):
    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)[0]
    prediction_prob = model.predict_proba(user_df)[0]

    if prediction == 1:
        st.error(f"The model predicts **Malignant (Cancer)**. Confidence: {prediction_prob[1]*100:.2f}%", icon="⚠️")
    else:
        st.success(f"The model predicts **Benign (No Cancer)**. Confidence: {prediction_prob[0]*100:.2f}%", icon="✅")
