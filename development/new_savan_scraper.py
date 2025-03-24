import requests
import concurrent.futures
from tqdm import tqdm
from datetime import date, timedelta, datetime
from bs4 import BeautifulSoup
import os
import random

class GamePKScraper:
    def __init__(self, start_dt: str, end_dt: str = None):
        self.GAMEDAY_RANGE_URL = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={}&endDate={}&timeZone=America/New_York&gameType=E&&gameType=S&&gameType=R&&gameType=F&&gameType=D&&gameType=L&&gameType=W&&gameType=A&&gameType=C&language=en&leagueId=103&&leagueId=104&hydrate=team,flags,broadcasts(all),venue(location)&sortBy=gameDate,gameStatus,gameType'
        self.start_dt = start_dt
        self.end_dt = end_dt

        if self.end_dt is None:
            self.end_dt = self.start_dt # Makes end date optional to get one day

        self.start_dt_date, self.end_dt_date = datetime.strptime(self.start_dt, "%Y-%m-%d"), datetime.strptime(self.end_dt, "%Y-%m-%d")
        self.MLB_SEASON_DATES = {2020: (datetime(2020, 5, 15), datetime(2020, 10, 30))} # CAN HAVE A VALID SEASON DATES HERE

    def extract_game_info(self):

        range = list(self._date_range(self.start_dt_date, self.end_dt_date))

        game_pks = []
        with tqdm(total=len(range)) as progress:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._get_game_pks, subq_start, subq_end) for subq_start, subq_end in range}
                for future in concurrent.futures.as_completed(futures):
                    game_pks.extend(future.result())
                    progress.update(1)
        return list(set(game_pks)) # Prevents duplicate game ids


    def _get_game_pks(self, start_dt: date, end_dt: date):
        start_dt, end_dt = datetime.strftime(start_dt, "%Y-%m-%d"), datetime.strftime(end_dt, "%Y-%m-%d")

        response = requests.get(self.GAMEDAY_RANGE_URL.format(start_dt, end_dt))
        game_pk_list = []

        if response.status_code == 200:
            dates = response.json()['dates']
            for games in dates:
                for game in games['games']:
                    game_pk = game.get('gamePk', None)
                    game_pk_list.append(game_pk)

            return game_pk_list
        
        else:
            print('Error with response, error code', response.status_code)
            return game_pk_list
    

    def _date_range(self, start_dt: date, stop: date, step: int = 1):
        low = start_dt

        while low <= stop:
            season_start, season_end = low.replace(month=3, day=15), low.replace(month=11, day=15) # Can change this
            
            if low < season_start:
                low = season_start

                print("Skipping Offseason Dates")

            elif low > season_end:
                low = date(month=3, day=15, year=low.year + 1)
            
            if low > stop:
                return

            high = min(low + timedelta(step-1), stop)

            yield low, high

            low +=timedelta(days=step)

class GamedayScraper(GamePKScraper):
    def __init__(self, start_dt: str, end_dt: str, 
                 player: int, team: str, 
                 pitch_type: str,
                 max_return_videos: int,
                 max_videos_per_game: int):
        
        super().__init__(start_dt=start_dt, end_dt=end_dt)
        self.GAMEDAY_URL = 'https://statsapi.mlb.com/api/v1/game/{}/playByPlay'
        self.player = player # Search based on player id
        self.team = team # Optional search based on team abbreviation, could be put in the GamePKScraper if wanted
        self.pitch_type = pitch_type
        self.game_pks = self.extract_game_info()
        self.max_return_videos = max_return_videos
        self.max_videos_per_game = max_videos_per_game


    def process_games(self):
        play_ids = []

        with tqdm(total=len(self.game_pks)) as progress:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._request_url, game_pk) for game_pk in self.game_pks}
                for future in concurrent.futures.as_completed(futures):
                    play_ids.extend(future.result())
                        
                    progress.update(1)
        if self.max_return_videos:
            return random.sample(play_ids, self.max_return_videos)
        return play_ids


    def _request_url(self, game_pk: int):
        response = requests.get(self.GAMEDAY_URL.format(game_pk))
        play_ids = []

        if response.status_code == 200:
            data = response.json()
            for play in data['allPlays']:
                batter = play['matchup']['batter']['id']
                pitcher = play['matchup']['pitcher']['id']
                for pitch in play.get('playEvents', {}):
                    if not pitch:
                        print('Skip Pitch, no data')
                        continue

                    play_id = pitch.get('playId', None)

                    if play_id is not None:
                        pitch_type = pitch.get('details').get('type')

                        if batter == self.player or pitcher == self.player or pitch_type == self.pitch_type:
                            play_ids.append(play_id)

                        else:
                            play_ids.append(play_id)
            if self.max_videos_per_game:
                return random.sample(play_ids, self.max_videos_per_game)
            return play_ids

        else:
            return
    
class BaseballSavScraper:
    def __init__(self, start_dt: str, end_dt: str = None, 
                 player: int = None, team: str = None, pitch_type: str = None,
                 download_folder: str = 'savant_videos', 
                 max_return_videos: int = 10, 
                 max_videos_per_game: int = None):
        
        self.play_ids = GamedayScraper(start_dt, end_dt, player, team, pitch_type, max_return_videos, max_videos_per_game).process_games()
        self.VIDEO_URL = 'https://baseballsavant.mlb.com/sporty-videos?playId={}'
        self.download_folder = download_folder
        self.max_return_videos = max_return_videos
        self.max_videos_per_game = max_videos_per_game
        os.makedirs(self.download_folder, exist_ok=True)

    def load_and_write_content(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self._download_videos, self.play_ids)
                

    def _download_videos(self, play_id):
        response = requests.get(self.VIDEO_URL.format(play_id))

        if response.status_code == 200:

            soup = BeautifulSoup(response.content, 'html.parser')

            video_container = soup.find('div', class_='video-box')
            if video_container:
                    video_url = video_container.find('video').find('source', type='video/mp4')['src']
                    with requests.Session().get(video_url, stream=True) as r:
                        self._write_content(play_id, r)
                    print('Successfully downloaded video', play_id)
        return None
    
    def _write_content(self, play_id, response):
        content_file = os.path.join(self.download_folder, f'{play_id}.mp4')
        with open(content_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)



if __name__ == '__main__':
    d = BaseballSavScraper('2024-06-02', '2024-06-02').load_and_write_content()
