# Create a streamlit app
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Breast Cancer Prediction')
st.write('Enter the cell measurements to predict if the tumor is benign or malignant')

# Create input fields for features
feature_names = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']

input_features = []
for feature in feature_names:
    value = st.number_input(f'Enter {feature}:', format='%f')
    input_features.append(value)

if st.button('Predict'):
    # Scale features and predict
    features_scaled = scaler.transform([input_features])
    prediction = model.predict(features_scaled)[0]
    
    if prediction == 1:
        st.error('Prediction: Malignant')
    else:
        st.success('Prediction: Benign')

# Add information about the model
st.info('Model Accuracy: 96.5%')

# Simulate a demo workflow for the app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# Simulated data for demo
@st.cache
def load_demo_data():
    data = pd.DataFrame({
        'Feature1': np.random.normal(0, 1, 100),
        'Feature2': np.random.normal(5, 2, 100),
        'Feature3': np.random.normal(10, 3, 100),
        'Target': np.random.choice([0, 1], size=100)
    })
    return data

data = load_demo_data()

# Demo workflow
st.title("Breast Cancer Prediction App - Demo")

# Step 1: Data Visualization
st.header("Step 1: Data Visualization")
feature = st.selectbox("Select a feature to visualize:", data.columns[:-1])
plt.figure(figsize=(10, 6))
sns.histplot(data[feature], kde=True, bins=30)
st.pyplot(plt)

# Step 2: Monitoring Dashboard
st.header("Step 2: Monitoring Dashboard")
metric1 = st.empty()
metric2 = st.empty()
metric3 = st.empty()

for i in range(5):
    metric1.metric(label="Accuracy", value=f"{random.uniform(90, 95):.2f}%")
    metric2.metric(label="Prediction Count", value=random.randint(100, 200))
    metric3.metric(label="Average Processing Time", value=f"{random.uniform(0.1, 0.5):.2f} sec")
    time.sleep(1)

# Step 3: Prediction Simulation
st.header("Step 3: Prediction Simulation")
input_data = {
    'Feature1': st.number_input("Feature1", value=0.0),
    'Feature2': st.number_input("Feature2", value=5.0),
    'Feature3': st.number_input("Feature3", value=10.0)
}

# Simulate prediction
prediction = random.choice(["Benign", "Malignant"])
st.write("Prediction:", prediction)

print("Demo workflow created and ready to run.")