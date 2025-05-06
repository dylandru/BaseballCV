from baseballcv.functions import BaseballSavVideoScraper
import cv2
import os

class DatasetCreator:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        BaseballSavVideoScraper('2024-05-10', self.output_folder, team_abbr='CLE', max_return_videos=1).run_executor()


    def generate_example_images(self):
        cap = cv2.VideoCapture(self.output_folder)

        while cap.isOpened():
            read, frame = cap.read()

            for i in range(5):
                file_name = os.path.join(self.output_folder, f"frame_{i}.jpg")
                cv2.imwrite(file_name, frame)
        
    def genereate_example_video(self):
        return self.output_folder