import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    try:
        df = pd.read_csv("Airport_Flight_Data_Final_Updated.csv")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Train the ML model
def train_model(df):
    if df is None:
        return None
    
    required_columns = ["Season", "Flight_Type", "Total_Flights", "Passengers"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        return None
    
    df = pd.get_dummies(df, columns=["Season", "Flight_Type"], drop_first=True)
    
    X = df.drop(columns=["Passengers"])
    y = df["Passengers"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    st.write(f"Model Trained! MAE: {mae}, MSE: {mse}")
    return model

# Predict future footfalls
def predict_future(model, df):
    if model is None or df is None:
        return None
    
    future_dates = pd.date_range(start=df["Date"].max(), periods=30, freq='D')
    future_df = pd.DataFrame({"Date": future_dates})
    future_df["Total_Flights"] = df["Total_Flights"].mean()
    
    future_df = pd.get_dummies(future_df, columns=["Season", "Flight_Type"], drop_first=True)
    
    for col in df.columns:
        if col not in future_df.columns and col != "Passengers":
            future_df[col] = 0
    
    future_passengers = model.predict(future_df.drop(columns=["Date"], errors='ignore'))
    future_df["Predicted_Passengers"] = future_passengers
    
    return future_df

# Main Streamlit App
df = load_data()
if df is not None:
    st.title("Airport Footfall Predictor")
    
    model = train_model(df)
    future_predictions = predict_future(model, df)
    
    if future_predictions is not None:
        st.subheader("Future Passenger Predictions")
        st.line_chart(future_predictions.set_index("Date")["Predicted_Passengers"])
