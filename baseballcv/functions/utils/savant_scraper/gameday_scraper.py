import random
import concurrent.futures
from tqdm import tqdm
import requests
from .gamepk_scraper import GamePKScraper


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