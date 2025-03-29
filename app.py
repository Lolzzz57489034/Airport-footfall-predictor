import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
def load_data():
    file_path = "Airport_Flight_Data_Final_Updated.csv"
    df = pd.read_csv(file_path)
    df.rename(columns=lambda x: x.strip(), inplace=True)  # Remove any unwanted spaces
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')  # Convert to datetime
    return df.dropna()

# Train ML Model
def train_model(df):
    features = ["Year", "Month", "Season", "Flight_Type", "Total_Flights"]
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df = pd.get_dummies(df, columns=["Season", "Flight_Type"], drop_first=True)
    
    X = df[features]
    y = df["Predicted_Footfall"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    st.sidebar.write(f"Model MAE: {mean_absolute_error(y_test, predictions):.2f}")
    return model

# Streamlit UI
st.title("✈️ Airport Footfall Prediction")
st.subheader("🔮 Predict Future Airport Footfall")

df = load_data()
model = train_model(df)

airports = df["Airport"].unique().tolist()
seasons = ["Winter", "Summer", "Monsoon"]
flight_types = df["Flight_Type"].unique().tolist()

airport = st.selectbox("Select Airport", airports)
season = st.selectbox("Select Season", seasons)
flight_type = st.selectbox("Select Flight Type", flight_types)
year = st.slider("Select Future Year", min_value=2025, max_value=2035, value=2026)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [6],  # Assuming prediction for mid-year
        "Season_Winter": [1 if season == "Winter" else 0],
        "Season_Summer": [1 if season == "Summer" else 0],
        "Flight_Type_International": [1 if flight_type == "International" else 0],
        "Total_Flights": [df[df["Airport"] == airport]["Total_Flights"].mean()]
    })
    
    predicted_footfall = model.predict(input_data)[0]
    st.success(f"Predicted Footfall for {airport} in {year}: {int(predicted_footfall)}")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    df_filtered = df[(df["Airport"] == airport) & (df["Flight_Type"] == flight_type)]
    df_filtered = df_filtered.groupby("Year")["Predicted_Footfall"].mean().reset_index()
    
    sns.lineplot(x=df_filtered["Year"], y=df_filtered["Predicted_Footfall"], marker='o', label="Historical Predictions")
    plt.axvline(year, color='r', linestyle='--', label=f"Predicted {year}")
    plt.xlabel("Year")
    plt.ylabel("Footfall")
    plt.title(f"Footfall Trend for {airport}")
    plt.legend()
    st.pyplot(plt)
