# Create a streamlit app
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Title and description
st.title("Cell Classification: Malignant or Benign")
st.write("""
This application trains a neural network to classify whether a cell is **malignant** or **benign**. 
You can upload a dataset, train the model, and see the results interactively.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    # Load the data
    df = pd.read_csv("/content/cleaned_data.csv")
    st.write("Dataset preview:")
    st.write(df.head())
    
    if 'Diagnosis' not in df.columns:
        st.error("The dataset must contain a 'Diagnosis' column for classification.")
    else:
        # Preprocess the data
        st.subheader("Model Training")
        with st.spinner("Preparing data..."):
            X = df.drop('Diagnosis', axis=1)
            y = df['Diagnosis'].replace({'yes': 1, 'no': 0})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        st.success("Data prepared successfully!")

        # Build and compile the model
        st.write("Building and training the model...")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification output
        ])
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

        # Train the model
        with st.spinner("Training the model..."):
            history = model.fit(
                X_train, y_train, 
                batch_size=128, 
                epochs=50, 
                verbose=0, 
                validation_data=(X_test, y_test)
            )

        st.success("Model trained successfully!")

        # Display training accuracy and loss
        st.subheader("Training Results")
        st.line_chart(pd.DataFrame({
            'Loss': history.history['loss'],
            'Validation Loss': history.history['val_loss']
        }))
        st.line_chart(pd.DataFrame({
            'Accuracy': history.history['accuracy'],
            'Validation Accuracy': history.history['val_accuracy']
        }))

        # Evaluate the model
        score = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"**Test Loss:** {score[0]:.4f}")
        st.write(f"**Test Accuracy:** {score[1]:.4f}")

        # Make predictions
        predictions = model.predict(X_test)
        predicted_classes = (predictions > 0.5).astype(int)

        # Show predictions
        st.subheader("Predictions on Test Data")
        results_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': predicted_classes.flatten()
        })
        st.write(results_df)

        # Allow user to input new data for prediction
        st.subheader("Make a New Prediction")
        new_data = {}
        for col in X.columns:
            new_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

        if st.button("Predict"):
            input_data = scaler.transform([list(new_data.values())])
            new_prediction = model.predict(input_data)
            prediction_class = "Malignant" if new_prediction > 0.5 else "Benign"
            st.write(f"The prediction is: **{prediction_class}**")
