from .crawler import Crawler
from datetime import datetime, date
import requests
from tqdm import tqdm
import concurrent.futures
import random
from typing import List

class GamePKScraper(Crawler):
    """
    Scraping Class that focuses on scraping the game ids based on a date range. Inherits from the Crawler class.
    """
    def __init__(self, start_dt: str, end_dt: str=None, team_abbr: str=None) -> None:
        super().__init__(start_dt, end_dt)
        self.GAMEDAY_RANGE_URL = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={}&endDate={}&timeZone=America/New_York&gameType=E&&gameType=S&&gameType=R&&gameType=F&&gameType=D&&gameType=L&&gameType=W&&gameType=A&&gameType=C&language=en&leagueId=103&&leagueId=104&hydrate=team,flags,broadcasts(all),venue(location)&sortBy=gameDate,gameStatus,gameType'
        self.team_abbr = team_abbr
        corrected_teams_dict = {'CHW': 'CWS',
                                'OAK': 'ATH',
                                'ARI': 'AZ' }
        self.team_abbr = corrected_teams_dict.get(self.team_abbr, self.team_abbr)

        # I believe these are all the correct abbreviations from MLB Gameday. I can check
        recognized_abbr = ['CLE', 'CHC', 'ATL', 'AZ', 'ATH', 'CWS', 
                                'LAD', 'LAA', 'BAL', 'BOS', 'CIN', 'COL', 'DET', 'HOU', 'KC',
                                'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'PHI', 'PIT', 'SD',
                                'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WAS']
        
        if self.team_abbr not in recognized_abbr:
            raise ValueError(f"""
            WARNING: Team Abbreviation {self.team_abbr} was not recognized. Please use proper team abbreviations. The following are converted
            for your convenience:
            * ARI -> AZ
            * OAK -> ATH
            * CHW -> CWS
            """)
        

    def run_executor(self) -> List[int]:
        range = list(self._date_range(self.start_dt_date, self.end_dt_date))

        game_pks = []
        with tqdm(total=len(range)) as progress:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._get_game_pks, subq_start, subq_end) for subq_start, subq_end in range}
                for future in concurrent.futures.as_completed(futures):
                    game_pks.extend(future.result())
                    progress.update(1)
        return list(set(game_pks)) # Prevents duplicate game ids

    
    def _get_game_pks(self, start_dt: date, end_dt: date) -> List[int]:
        """
        Function that gets the game ids within each corresponding link.

        Parameters:
            start_dt (date): The start date of the query.
            end_dt (date): The end date of the query.

        Returns:
            List[int]: A list of the game ids unless an invalid status code.
        """
        self.rate_limiter()
        start_dt, end_dt = datetime.strftime(start_dt, "%Y-%m-%d"), datetime.strftime(end_dt, "%Y-%m-%d")

        response = requests.get(self.GAMEDAY_RANGE_URL.format(start_dt, end_dt))
        game_pk_list = []

        if response.status_code == 200:
            dates = response.json()['dates']
            for games in dates:
                for game in games['games']:
                    home_team = game['teams']['home']['team'].get('abbreviation', 'Unknown')
                    away_team = game['teams']['away']['team'].get('abbreviation', 'Unknown')
                    game_pk = game.get('gamePk', None)

                    if home_team == self.team_abbr or away_team == self.team_abbr:
                        game_pk_list.append(game_pk)
                    elif self.team_abbr is None:
                        game_pk_list.append(game_pk)
            return game_pk_list
        
        else:
            print('Error with response, error code', response.status_code)
            return game_pk_list
        
class GamePlayIDScraper(GamePKScraper):
    """
    Class that extracts the play ids for each game. Inherits from the GamePKScraper class.
    """

    def __init__(self, start_dt, end_dt=None, team_abbr=None, **kwargs) -> None:
        super().__init__(start_dt, end_dt, team_abbr)
        self.GAMEDAY_URL = 'https://statsapi.mlb.com/api/v1/game/{}/playByPlay'
        self.player = kwargs.get('player', None)
        self.pitch_type = kwargs.get('pitch_type', None)
        self.max_return_videos = kwargs.get('max_return_videos', 10)
        self.max_videos_per_game = kwargs.get('max_videos_per_game', None)

        self.game_pks = super().run_executor()
        
        if not self.game_pks:
            raise ValueError(f"Cannot Scrape Game IDs with no Game IDs. No games played from {str(start_dt)} to {str(end_dt)}")
        
        if self.player and not team_abbr:
            print("Warning, this may run slower as it's looking for all teams. Please consider using team abbreviation in addition to player id to make the extraction faster.")
       
        
    def run_executor(self) -> List[int]:
        play_ids = []

        with tqdm(total=len(self.game_pks)) as progress:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._get_play_ids, game_pk) for game_pk in self.game_pks}
                for future in concurrent.futures.as_completed(futures):
                    play_ids.extend(future.result())
                        
                    progress.update(1)
        if self.max_return_videos:
            play_id_len = len(play_ids)
            return random.sample(play_ids, min(self.max_return_videos, play_id_len)) # Prevents error if the max videos is larger than the actual return videos.
        return play_ids
    
    def _get_play_ids(self, game_pk: int) -> List[int] | None:
        """
        Function that extracts tha play ids for each game.

        Parameters:
            game_pk (int): The game id.
        
        Returns:
            List[int] | None: A list of the play ids or None if the status code is invalid.
        """

        self.rate_limiter()

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
                        pitch_type = pitch['details'].get('type')

                        if pitch_type is not None:
                            pitch_type = pitch_type.get('code')
                        
                        # My brain was fried after this
                        if (batter == self.player or pitcher == self.player) and pitch_type == self.pitch_type:
                            play_ids.append(play_id)

                        elif (batter == self.player or pitcher == self.player) and self.pitch_type is None:
                            play_ids.append(play_id)

                        elif self.player is None and pitch_type == self.pitch_type:
                            play_ids.append(play_id)

                        elif self.player is None and self.pitch_type is None:
                            play_ids.append(play_id)

            if self.max_videos_per_game:
                return random.sample(play_ids, self.max_videos_per_game)
            return play_ids

        else:
            return