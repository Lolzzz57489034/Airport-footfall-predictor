import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Dataset
def load_data():
    df = pd.read_csv("Airport_Flight_Data_Final_Updated.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df.dropna(subset=["Date"], inplace=True)
    df["Year"] = df["Date"].dt.year.astype(int)
    return df

df = load_data()

# Streamlit UI
st.title("\U0001F6EB Airport Footfall Prediction")
st.subheader("\U0001F52E Predict Future Airport Footfall")

# Airport Selection
airports = df["Airport"].unique().tolist()
selected_airport = st.selectbox("Select Airport", airports)

# Season Selection
seasons = ["Winter", "Summer", "Monsoon"]
selected_season = st.selectbox("Select Season", seasons)

# Flight Type Selection
flight_types = ["Domestic", "International"]
selected_flight_type = st.selectbox("Select Flight Type", flight_types)

# Predict Year
future_year = st.slider("Select Future Year:", 2025, 2034, 2026)

# Filter Data
filtered_df = df[(df["Airport"] == selected_airport) & 
                 (df["Season"] == selected_season) & 
                 (df["Flight_Type"] == selected_flight_type)]

# Model Training
X = filtered_df[["Year"]]
y = filtered_df["Predicted_Footfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
future_prediction = model.predict([[future_year]])[0]
st.write(f"Predicted footfall for {selected_airport} in {future_year}: {future_prediction:.2f}")

# Visualization
plt.figure(figsize=(10, 5))
sns.lineplot(x=X["Year"], y=y, marker='o', label="Actual Footfall")
sns.lineplot(x=[future_year], y=[future_prediction], marker='o', color='red', label="Predicted Footfall")
plt.xlabel("Year")
plt.ylabel("Footfall")
plt.title(f"Footfall Prediction for {selected_airport}")
plt.legend()
st.pyplot(plt)
