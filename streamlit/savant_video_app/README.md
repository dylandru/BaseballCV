# Baseball Savant Data Tool

This Streamlit application allows users to fetch detailed pitch-by-pitch data directly from the Baseball Savant Statcast API.

## How to Run

1.  **Navigate to the app directory:**
    ```bash
    # From the root of the BaseballCV repository
    cd streamlit/savant_video_app
    ```

2.  **Ensure all dependencies are installed:**
    * Make sure you have an active virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will open in your default web browser.

## Features

-   **Direct API Querying:** Fetches data directly from the Statcast API for speed and reliability, avoiding HTML scraping.
-   **No External Dependencies:** Does not require `pybaseball` or `selenium`.
-   **Two Query Modes:**
    -   Search for plays using a wide range of filters (date, pitch type, player name, team, etc.).
    -   Look up a single, specific play using its `game_pk`, `at_bat_number`, and `pitch_number`.
-   **Data Export:** Download the full results of any search to a CSV file.
-   **Film Room Links:** Provides a direct link to the "Film Room" page for each play.
