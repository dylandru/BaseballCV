import os

def test_run_statcast_pull_scraper(scraper):
    """
    Tests the main video scraping functionality.
    Uses yesterday's date to ensure game data exists and limits to 2 videos 
    total to keep the test quick while still verifying functionality.
    """
    #Run Scraper for Yankees on 3/28/2024 w/ minimal videos
    df = scraper.run_statcast_pull_scraper(
        start_date="2024-03-28",
        end_date="2024-03-28",
        download_folder="test_videos",
        max_workers=2,
        max_videos=2,
        max_videos_per_game=1,
        team="NYY"
    )
    
    assert not df.empty, "Should return data"
    assert os.path.exists("test_videos"), "Download folder should exist"
    videos = os.listdir("test_videos")
    assert len(videos) > 0 and videos[0].endswith(".mp4"), "Should have downloaded videos"
    scraper.cleanup_savant_videos("test_videos")