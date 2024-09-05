import os
from collections import defaultdict
import concurrent.futures
from function_utils import clone_savant_video_scraper, extract_frames_from_video, cleanup_savant_videos

def generate_photo_dataset(output_frames_folder: str = "cv_dataset", 
                           video_download_folder: str = "raw_videos",
                           max_plays: int = 10, 
                           max_num_frames: int = 6000,
                           start_date: str = "2024-05-22",
                           end_date: str = "2024-05-25",
                           max_workers: int = 10,
                           delete_savant_videos: bool = True) -> None:
    """
    Extracts random frames for a given number of plays from a folder of scraped Baseball Savant broadcast videos to create a photo dataset for a 
    Computer Vision model.
    
    Args:
        output_frames_folder (str): Name of folder where photos will be saved. Default is "cv_dataset".
        video_download_folder (str): Name of folder containing videos. Default is "raw_videos".
        max_plays (int): Maximum number of plays for scraper to download videos. Default is 10.
        max_num_frames (int): Maximum number of frames to extract across all videos. May be less if not enough frames exist from plays. Default 
                              is 6000.
        start_date (str): Start date for video scraping in "YYYY-MM-DD" format. Default is "2024-05-22".
        end_date (str): End date for video scraping in "YYYY-MM-DD" format. Default is "2024-05-25".
        max_workers (int): Number of worker processes to use for frame extraction. Default is 10.
        delete_savant_videos (bool): Whether or not to delete scraped savant videos after frames are extracted. Default is True.

    Returns:
        None: Returns nothing. Creates a folder of photos from the videos frames to use.
    """

    clone_savant_video_scraper()

    from BSav_Scraper_Vid.MainScraper import run_statcast_pull_scraper

    run_statcast_pull_scraper(start_date=start_date, end_date=end_date, 
                              download_folder=video_download_folder, max_videos=max_plays)
            
    os.makedirs(output_frames_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_download_folder) if f.endswith(('.mp4', '.mov'))]
    
    if not video_files:
        print("No video files found in the specified folder.") #ensure video files are downloaded from scraper
        return
    
    games = defaultdict(list) #group videos by given game for increased variety
    for video_file in video_files:
        game_id = video_file[:6] 
        games[game_id].append(video_file)
    
    frames_per_game = max_num_frames // len(games)
    remaining_frames = max_num_frames % len(games) #distribute frames evenly across games
    
    extraction_tasks = []
    for game_id, game_videos in games.items():
        frames_for_game = frames_per_game + (1 if remaining_frames > 0 else 0)
        remaining_frames = max(0, remaining_frames - 1)
        
        frames_per_video = frames_for_game // len(game_videos)
        extra_frames = frames_for_game % len(game_videos)
        
        for i, video_file in enumerate(game_videos):
            frames_to_extract = frames_per_video + (1 if i < extra_frames else 0)
            video_path = f"{video_download_folder}/{video_file}"
            extraction_tasks.append((video_path, game_id, output_frames_folder, frames_to_extract))
    
    extracted_frames = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_videos = {executor.submit(extract_frames_from_video, *task): task for task in extraction_tasks}
        for future in concurrent.futures.as_completed(future_videos):
            video_path, game_id, _, _ = future_videos[future]
            try:
                result = future.result()
                extracted_frames.extend(result)
            except Exception as e:
                print(f"Error with {video_path}: {str(e)}")

    print(f"Extracted {len(extracted_frames)} frames from {len(video_files)} videos from {len(games)} games. Videos now deleting...")
    
    if delete_savant_videos:
        cleanup_savant_videos(video_download_folder)

    else:
        return None
    

        

