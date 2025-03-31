import streamlit as st
import pandas as pd

# Load dataset to extract unique values for dropdowns
file_path = "Airport_Flight_Data_Final_Updated.csv"
df = pd.read_csv(file_path)

# Extract unique values
airports = df["Airport"].unique().tolist()
seasons = ["Summer", "Monsoon", "Winter"]
flight_types = ["Domestic", "International"]
years = sorted(df["Year"].unique().tolist())
weekday_options = ["Weekday", "Weekend"]

# Streamlit UI
st.title("Airport Footfall Prediction")
st.sidebar.header("Input Parameters")

selected_airport = st.sidebar.selectbox("Select Airport:", airports)
selected_season = st.sidebar.selectbox("Select Season:", seasons)
selected_flight_type = st.sidebar.selectbox("Select Flight Type:", flight_types)
selected_year = st.sidebar.slider("Select Year:", min_value=min(years), max_value=max(years), step=1)
selected_weekday = st.sidebar.radio("Flight Day:", weekday_options)

# Display selected inputs
st.write("### Selected Inputs")
st.write(f"**Airport:** {selected_airport}")
st.write(f"**Season:** {selected_season}")
st.write(f"**Flight Type:** {selected_flight_type}")
st.write(f"**Year:** {selected_year}")
st.write(f"**Day Type:** {selected_weekday}")
