import pandas as pd
import streamlit as st
from io import StringIO

@st.cache_data(show_spinner="Loading player ID map...")
def load_player_id_map() -> pd.DataFrame:
    """
    Loads the player ID mapping from the public Google Sheet.
    The result is cached to prevent re-downloading on every script run.

    Returns:
        pd.DataFrame: A DataFrame containing player names and their MLBAM IDs.
    """
    try:
        # The URL for the Google Sheet, published as a CSV
        sheet_url = "https://docs.google.com/spreadsheets/d/1JgczhD5VDQ1EiXqVG-blttZcVwbZd5_Ne_mefUGwJnk/export?format=csv&gid=0"
        df = pd.read_csv(sheet_url)
        
        # We only need the name and ID columns
        df = df[['MLBNAME', 'MLBID']].copy()

        # FIX: Ensure all player names are treated as strings to prevent sorting errors
        df['MLBNAME'] = df['MLBNAME'].astype(str)

        # Clean the data: remove rows with missing IDs or names
        df.dropna(subset=['MLBID', 'MLBNAME'], inplace=True)
        df['MLBID'] = df['MLBID'].astype(int)
        df.rename(columns={'MLBNAME': 'name', 'MLBID': 'id'}, inplace=True)

        print("DEBUG: Player ID map loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Failed to load player ID map from Google Sheets: {e}")
        return pd.DataFrame()

def get_player_id(player_name: str, player_df: pd.DataFrame) -> int:
    """
    Finds the MLBID for a given player name from the provided DataFrame.
    """
    if player_df.empty or not player_name:
        return 0
    
    # Simple case-insensitive match first
    match = player_df[player_df['name'].str.lower() == player_name.lower()]
    
    if not match.empty:
        return match.iloc[0]['id']
        
    # If no exact match, we can implement fuzzy matching in the future
    # For now, we rely on the selectbox providing the exact name.
    return 0