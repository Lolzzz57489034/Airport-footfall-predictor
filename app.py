@st.cache_data
def load_data():
   df = pd.read_csv(file_path)

# Check if 'Date' column exists
if "Date" not in df.columns:
    st.error("Error: 'Date' column not found in the dataset.")
    st.stop()

# Display first few values
st.write("### First few values in 'Date' column:")
st.write(df["Date"].head(10))

# Check data type
st.write(f"Data type of 'Date' column: {df['Date'].dtype}")
