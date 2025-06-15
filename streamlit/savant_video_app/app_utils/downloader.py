import os
import io
import zipfile
import streamlit as st
import pandas as pd
import yt_dlp

def create_zip_in_memory(selected_rows: pd.DataFrame):
    """
    Fetches videos using yt-dlp and stores them in a zip file in memory.

    Args:
        selected_rows (pd.DataFrame): DataFrame of rows selected by the user.

    Returns:
        BytesIO: A bytes object representing the zip file.
    """
    zip_buffer = io.BytesIO()

    total_videos = len(selected_rows)
    progress_bar = st.progress(0, text="Initializing download...")

    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, row in enumerate(selected_rows.itertuples()):
            try:
                # Update progress bar text
                progress_text = f"Downloading video {i+1} of {total_videos}: {row.batter_name} vs {row.pitcher_name}"
                progress_bar.progress((i + 1) / total_videos, text=progress_text)

                film_room_url = row.video_url
                
                # FIX: Ensure player names are strings before trying to replace characters.
                # If a name wasn't found, it might be an integer ID.
                batter_str = str(row.batter_name).replace(' ', '_')
                pitcher_str = str(row.pitcher_name).replace(' ', '_')
                
                # Create a descriptive filename for the video
                filename = f"{row.game_date}_{batter_str}_vs_{pitcher_str}_{row.play_id[:8]}.mp4"

                # Use a temporary dictionary to hold video data for yt-dlp
                buffer = io.BytesIO()
                
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'outtmpl': '-', # Directs output to stdout
                    'logtostderr': True, # Suppresses console output
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                }

                # --- This is a workaround to capture stdout ---
                # yt-dlp doesn't have a clean way to pipe to a buffer in memory,
                # so we will use a small temporary file as an intermediate step.
                temp_filename = f"temp_{row.play_id}.mp4"
                ydl_opts['outtmpl'] = temp_filename
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([film_room_url])

                # Read from temp file and write to zip
                if os.path.exists(temp_filename):
                    with open(temp_filename, 'rb') as f:
                        zip_file.writestr(filename, f.read())
                    os.remove(temp_filename) # Clean up the temp file
                    print(f"DEBUG: Successfully added {filename} to zip.")

            except Exception as e:
                st.error(f"An error occurred for playId {row.play_id}: {e}")
                # Clean up temp file in case of error
                if 'temp_filename' in locals() and os.path.exists(temp_filename):
                    os.remove(temp_filename)

    progress_bar.empty() # Remove progress bar when done
    return zip_buffer