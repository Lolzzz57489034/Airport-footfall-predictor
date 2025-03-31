@st.cache_data
def load_data():
    df = pd.read_csv(file_path)

    # Check if 'Date' column exists
    if "Date" not in df.columns:
        st.error("Error: 'Date' column not found in the dataset.")
        return None

    # Print first few values for debugging
    st.write("### Sample values from 'Date' column:")
    st.write(df["Date"].head(10))

    # Try converting Date column
    try:
        df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d/%m/%Y", errors="coerce")
    except Exception as e:
        st.error(f"Date conversion failed: {e}")
        return None

    # Remove rows with invalid dates
    df = df.dropna(subset=["Date"])

    return df
