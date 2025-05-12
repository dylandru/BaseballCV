from baseballcv.functions import BaseballSavVideoScraper
import cv2
import os
import random
from datetime import date
import streamlit as st

class DatasetCreator:
    def __init__(self):
         pass

    def generate_video(self, output_video):
        teams = [
                "ATH", "ATL", "AZ", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET", "HOU",
                "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "PHI", "PIT", "SD", "SEA",
                "SF", "STL", "TB", "TEX", "TOR", "WSH"
            ]
        
        random_team = random.choice(teams)
        random_date = date.strftime(date(random.randint(2021, 2024), random.randint(4, 8), random.randint(1, 25)), '%Y-%m-%d')
        
        BaseballSavVideoScraper(random_date, download_folder=output_video, team_abbr=random_team, max_return_videos=1).run_executor()
        

    def generate_example_images(self, img_file, cap: cv2.VideoCapture, length: int, *args):
        for arg in args:
            if callable(arg):
                styling = arg
                styling()
                break
            
        random_index = random.sample(range(0, length), 3)

        i = 0
        while cap.isOpened():
            read, frame = cap.read()

            if read and i in random_index:
                file_name = os.path.join(img_file, f"frame_{i}.jpg")
                cv2.imwrite(file_name, frame)

            i += 1

            if i > max(random_index):
                break
