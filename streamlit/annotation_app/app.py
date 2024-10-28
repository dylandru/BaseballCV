import streamlit as st
import pandas as pd

# Sample DataFrame
data = {
    'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5'],
    'Position': ['Pitcher', 'Catcher', 'Hitter', 'Pitcher', 'Hitter'],
    'Average': [0.256, 0.300, 0.275, 0.280, 0.250],
}
df = pd.DataFrame(data)

# Sidebar filters
st.sidebar.header("Filter Options")
selected_position = st.sidebar.selectbox("Select Position", options=["All"] + df['Position'].unique().tolist())
filter_button = st.sidebar.button("Apply Filter")

# Filter the DataFrame
if filter_button:
    if selected_position != "All":
        filtered_df = df[df['Position'] == selected_position]
    else:
        filtered_df = df
else:
    filtered_df = df

# Show interactive table
st.header("Interactive Player Table")
st.dataframe(filtered_df)

# Adding actions (e.g., delete rows)
st.subheader("Row Actions")
for index, row in filtered_df.iterrows():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"{row['Player']} - {row['Position']} - Avg: {row['Average']}")
    with col2:
        if st.button(f"Delete {row['Player']}", key=index):
            df.drop(index, inplace=True)
            st.experimental_rerun()  # Rerun the app to refresh the table after deletion
