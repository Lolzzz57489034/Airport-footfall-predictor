import streamlit as st
import pandas as pd

# Load dataset
file_path = "Airport_Flight_Data_Final_Updated.csv"  # Ensure the correct path
@st.cache_data
def load_data():
    return pd.read_csv(file_path)

df = load_data()

# Streamlit App Title
st.title("Airport Footfall Prediction")

# User Inputs
st.sidebar.header("Select Parameters")

# Select Airport
airports = df["Airport"].unique()
selected_airport = st.sidebar.selectbox("Select Airport", airports)

# Select Season
seasons = ["Summer", "Monsoon", "Winter"]
selected_season = st.sidebar.selectbox("Select Season", seasons)

# Select Flight Type
flight_types = ["Domestic", "International"]
selected_flight_type = st.sidebar.selectbox("Select Flight Type", flight_types)

# Select Year for Prediction
min_year = df["Year"].min()
max_year = df["Year"].max()
predicted_year = st.sidebar.slider("Select Year for Prediction", min_year, max_year + 10, max_year + 1)

# Display Selected Parameters
st.write("### Selected Parameters")
st.write(f"**Airport:** {selected_airport}")
st.write(f"**Season:** {selected_season}")
st.write(f"**Flight Type:** {selected_flight_type}")
st.write(f"**Year for Prediction:** {predicted_year}")

st.write("Columns in dataset:", df.columns.tolist())


