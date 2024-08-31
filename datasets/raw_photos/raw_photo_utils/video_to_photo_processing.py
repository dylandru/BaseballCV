import os
import subprocess
import fileinput

def clone_savant_video_scraper() -> None:
    '''
    Clones the Baseball Savant Video Scraper into the repo directory and updates the import statement to reference the directory correctly.
    '''
    try:
        sav_url = "https://github.com/dylandru/BSav_Scraper_Vid.git"
        repo_name = sav_url.split('/')[-1].replace('.git', '')
        
        if os.path.exists(repo_name):
            print(f"Repository '{repo_name}' already exists.")
            return
        
        subprocess.run(['git', 'clone', sav_url], check=True)

        main_scraper_path = os.path.join(repo_name, 'MainScraper.py')
        
        with fileinput.FileInput(main_scraper_path, inplace=True) as file:
            for line in file:
                print(line.replace(
                    "from savant_video_utils import", 
                    "from .savant_video_utils import" #adds reference to parent directory to ensure module is identified correctly in running script within repo
                    ), end='')
        
        print(f"Repository '{repo_name}' cloned successfully with update.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cloning the repository: {e}")
        return None

