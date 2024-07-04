import streamlit as st
import pandas as pd

# Sample data (you can replace this with your actual data)
data = {
    'Clause': ['Clause A', 'Clause B', 'Clause C'],
    'Full Filled': ['✅', '❌', '✅'],
    'Evidence': ['Document 1', 'None', 'Document 2']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

st.title("Table Example")
st.caption("Example of creating a table in Streamlit with tick and cross emojis")

# Display the table using st.table
st.table(df)
