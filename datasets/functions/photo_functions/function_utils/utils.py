import subprocess
import fileinput
import os
import cv2
import random

def extract_frames_from_video(video_path, game_id, output_frames_folder, frames_to_extract) -> list[str]:
    '''
    Extracts frames from a single video file and saves the images into the specified folder.
    '''
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video {os.path.basename(video_path)}...")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_video = min(frames_to_extract, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), frames_video))
    
    extracted_frames = []
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            output_file = os.path.join(output_frames_folder, f"{game_id}_{os.path.splitext(os.path.basename(video_path))[0]}_{i:04d}.jpg")
            cv2.imwrite(output_file, frame)
            extracted_frames.append(output_file)
        else:
            print(f"Could not read frame {frame_index} from video {video_path}")
    
    cap.release()

    return extracted_frames

def clone_savant_video_scraper(dir: str ='datasets/functions/photo_functions') -> None:
    '''
    Clones the Baseball Savant Video Scraper into a specific directory and updates the import statement to reference the directory correctly inside of
    this repository.
    '''
    try:
        sav_url = "https://github.com/dylandru/BSav_Scraper_Vid.git"
        repo_name = sav_url.split('/')[-1].replace('.git', '')
        
        if not dir:
            dir = os.getcwd()
        
        clone_path = f"{dir}/{repo_name}"
        
        if os.path.exists(clone_path):
            print(f"Repository '{repo_name}' already exists in the specified directory.")
            return None
        
        subprocess.run(['git', 'clone', sav_url, clone_path], check=True)

        main_scraper_path = os.path.join(clone_path, 'MainScraper.py')
        
        with fileinput.FileInput(main_scraper_path, inplace=True) as file:
            for line in file:
                print(line.replace(
                    "from savant_video_utils import", 
                    "from .savant_video_utils import" #adds reference to parent directory to ensure module is identified correctly in running script within repo
                    ), end='')
        
        print(f"Repository '{repo_name}' cloned successfully with update.")
    except subprocess.CalledProcessError as e:
        print(f"Error while cloning repo: {e}")
        return None