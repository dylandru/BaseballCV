import os
import io
import zipfile
import streamlit as st
import pandas as pd
import yt_dlp

def create_zip_in_memory(selected_rows: pd.DataFrame):
    """
    Fetches videos using yt-dlp and stores them in a zip file in memory.
    """
    zip_buffer = io.BytesIO()
    total_videos = len(selected_rows)
    progress_bar = st.progress(0, text="Initializing download...")
    
    # Placeholder for warnings to show them all at the end
    warnings = []

    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, row in enumerate(selected_rows.itertuples()):
            temp_filename = "" # Initialize to prevent reference before assignment error
            try:
                progress_text = f"Downloading video {i+1}/{total_videos}: {row.batter_name} vs {row.pitcher_name}"
                progress_bar.progress((i + 1) / total_videos, text=progress_text)
                
                film_room_url = row.video_url
                batter_str = str(row.batter_name).replace(' ', '_')
                pitcher_str = str(row.pitcher_name).replace(' ', '_')
                filename = f"{row.game_date}_{batter_str}_vs_{pitcher_str}_{row.play_id[:8]}.mp4"
                
                temp_filename = f"temp_{row.play_id}.mp4"
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'outtmpl': temp_filename,
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([film_room_url])

                if os.path.exists(temp_filename):
                    with open(temp_filename, 'rb') as f:
                        zip_file.writestr(filename, f.read())
                    print(f"DEBUG: Successfully added {filename} to zip.")
                else:
                    # This case can happen if yt-dlp fails silently.
                    warnings.append(f"Could not retrieve video for playId {row.play_id}. It might be unavailable.")

            # FIX: Specifically catch the DownloadError from yt-dlp
            except yt_dlp.utils.DownloadError as e:
                if "Unsupported URL" in str(e):
                    warnings.append(f"Video for playId `{row.play_id}` is unavailable (Unsupported URL).")
                else:
                    warnings.append(f"A download error occurred for playId `{row.play_id}`.")
                print(f"DEBUG: yt-dlp download error for {row.play_id}: {e}")
            
            except Exception as e:
                warnings.append(f"An unexpected error occurred for playId `{row.play_id}`.")
                print(f"DEBUG: General error for {row.play_id}: {e}")

            finally:
                # Always clean up the temp file if it exists
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

    progress_bar.empty()
    
    # Display all collected warnings at the end
    for warning_text in warnings:
        st.warning(warning_text, icon="⚠️")

    return zip_buffer