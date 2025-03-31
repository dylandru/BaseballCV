from .crawler import Crawler
from datetime import datetime, date
from tqdm import tqdm
import concurrent.futures
from typing import List
import polars as pl
import logging

class GamePKScraper(Crawler):
    """
    Scraping Class that focuses on scraping the game ids based on a date range. Inherits from the `Crawler` class.
    """
    def __init__(self, start_dt: str, end_dt: str=None, team_abbr: str=None, player: int=None, logger: logging.Logger = None) -> None:
        self.logger = logger if logger else logging.getLogger(__name__)
        super().__init__(start_dt, end_dt, self.logger)
        self.GAMEDAY_RANGE_URL = 'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={}&endDate={}&timeZone=America/New_York&gameType=E&&gameType=S&&gameType=R&&gameType=F&&gameType=D&&gameType=L&&gameType=W&&gameType=A&&gameType=C&language=en&leagueId=103&&leagueId=104&hydrate=team,flags,broadcasts(all),venue(location)&sortBy=gameDate,gameStatus,gameType'
        self.home_team, self.away_team, self.player = [], [], player
        self.corrected_teams_dict = {'CHW': 'CWS',
                                'OAK': 'ATH',
                                'ARI': 'AZ' }
        self.team_abbr = self.corrected_teams_dict.get(team_abbr, team_abbr)

        # Team: Team ID
        self.recognized_abbr = {'ATH': 133, 'PIT': 134, 'SD': 135, 'SEA': 136,
                           'SF': 137, 'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141,
                           'MIN': 142, 'PHI': 143, 'ATL': 144, 'CWS': 145, 'MIA': 146,
                           'NYY': 147, 'MIL': 158, 'LAA': 108, 'AZ': 109, 'BAL': 110, 
                           'BOS': 111, 'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 
                           'DET': 116, 'HOU': 117, 'KC': 118, 'LAD': 119, 'WSH': 120,
                           'NYM': 121}
        if self.player:
            year = self.end_dt[0:4] # First 4 elements to get the year 
            url = f'https://statsapi.mlb.com/api/v1/sports/1/players?season={year}'
            response = self.requests_with_retry(url)
            people = response.json()['people']
            team_id = -100 # Dummy variable that should change if team id is found.

            for player in people:
                if player.get('id') == self.player:
                    team_id = player.get('currentTeam')['id']
                    break
            
            if team_id == -100:
                raise ValueError(f"Cannot find player ID {self.player}. Maybe a typo?")

            self.team_abbr = {v: k for k,v in self.recognized_abbr.items()}.get(team_id)

        if team_abbr != None and self.team_abbr not in self.recognized_abbr:
            raise ValueError(f"""
            ERROR: Team Abbreviation {self.team_abbr} was not recognized. Please use proper team abbreviations. The following are converted
            for your convenience:
            * ARI -> AZ
            * OAK -> ATH
            * CHW -> CWS
            """)
        
        if team_abbr != None and self.team_abbr != team_abbr:
            raise ValueError(f"Wrong abbreviation {team_abbr}. Perhaps you meant {self.team_abbr} for your player?")

    def run_executor(self) -> List[int]:
        range = list(self._date_range(self.start_dt_date, self.end_dt_date))

        game_pks = []
        with tqdm(total=len(range), desc="Extracting Game IDs from Dates") as progress:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._get_game_pks, subq_start, subq_end) for subq_start, subq_end in range}
                for future in concurrent.futures.as_completed(futures):
                    game_pks.extend(future.result())
                    progress.update(1)
                    
                    
        return list(set(game_pks)) # Prevents duplicate game ids

    
    def _get_game_pks(self, start_dt: date, end_dt: date) -> List[int]:
        """
        Function that gets the game ids within each corresponding link.

        Args:
            start_dt (date): The start date of the query.
            end_dt (date): The end date of the query.

        Returns:
            List[int]: A list of the game ids unless an invalid status code.
        """
        self.rate_limiter()
        start_dt, end_dt = datetime.strftime(start_dt, "%Y-%m-%d"), datetime.strftime(end_dt, "%Y-%m-%d")
        response = self.requests_with_retry(self.GAMEDAY_RANGE_URL.format(start_dt, end_dt))
        game_pk_list = []
        
        for games in response.json()['dates']:
            for game in games['games']:
                home_team = game['teams']['home']['team'].get('abbreviation', 'Unknown')
                away_team = game['teams']['away']['team'].get('abbreviation', 'Unknown')
                game_pk = game.get('gamePk', None)
                
                if self.team_abbr is None or home_team == self.team_abbr or away_team == self.team_abbr:
                    game_pk_list.append(game_pk)
                    self.home_team.append(home_team)
                    self.away_team.append(away_team)
        return game_pk_list

class GamePlayIDScraper(GamePKScraper):
    """
    Class that extracts the play ids for each game. Inherits from the `GamePKScraper` class.
    """

    def __init__(self, start_dt, end_dt=None, team_abbr=None, player = None, logger: logging.Logger = None, **kwargs) -> None:
        self.logger = logger if logger else logging.getLogger(__name__)
        super().__init__(start_dt, end_dt, team_abbr, player, self.logger)
        self.GAMEDAY_URL = 'https://statsapi.mlb.com/api/v1/game/{}/playByPlay'
        self.pitch_type = kwargs.get('pitch_type', None)
        self.max_return_videos = kwargs.get('max_return_videos', 10)
        self.max_videos_per_game = kwargs.get('max_videos_per_game', None)

        self.game_pks = super().run_executor()  

        self.rename_dict = {
                            'game_pk': 'game_pk',
                            'home_team': 'home_team',
                            'away_team': 'away_team',
                            'batter': 'batter',
                            'pitcher': 'pitcher',
                            'inning': 'inning',
                            'inning_top_bot': 'inning_top_bot',
                            'p_throws': 'p_throws',
                            'isPitch': 'is_pitch',
                            'count_balls': 'balls',
                            'count_strikes': 'strikes',
                            'count_outs': 'outs',
                            'playId': 'play_id',
                            'pitchNumber': 'pitch_number_ab',
                            'details_type_code': 'pitch_type',
                            'details_type_description': 'pitch_name',
                            'pitchData_strikeZoneTop': 'sz_top',
                            'pitchData_strikeZoneBottom': 'sz_bot',
                            'pitchData_coordinates_aX': 'ax',
                            'pitchData_coordinates_aY': 'ay',
                            'pitchData_coordinates_aZ': 'az',
                            'pitchData_coordinates_pfxX': 'pfx_x',
                            'pitchData_coordinates_pfxZ': 'pfx_z',
                            'pitchData_coordinates_pX': 'plate_x',
                            'pitchData_coordinates_pZ': 'plate_z',
                            'pitchData_coordinates_vX0': 'vx0',
                            'pitchData_coordinates_vY0': 'vy0',
                            'pitchData_coordinates_vZ0': 'vz0',
                            'pitchData_coordinates_x0': 'x0',
                            'pitchData_coordinates_y0': 'y0',
                            'pitchData_coordinates_z0': 'z0',
                            'pitchData_breaks_breakAngle': 'break_angle',
                            'pitchData_breaks_breakLength': 'break_length',
                            'pitchData_breaks_breakY': 'break_y',
                            'pitchData_breaks_breakHorizontal': 'horizontal_break',
                            'pitchData_breaks_breakVertical': 'vertical_break',
                            'pitchData_breaks_breakVerticalInduced': 'induced_vertical_break',
                            'pitchData_breaks_spinRate': 'spin_rate',
                            'pitchData_breaks_spinDirection': 'spin_direction',
                            'pitchData_zone': 'zone',
                            'pitchData_typeConfidence': 'pitchtype_confidence',
                            'pitchData_plateTime': 'plate_time',
                            'pitchData_extension': 'extension'
                        }
        
        if not self.game_pks:
            raise ValueError(f"Cannot Scrape Game DataFrames with no Game IDs. No games played from {str(start_dt)} to {str(end_dt)}")
        
        if self.player and not team_abbr:
            self.logger.warning(f"Warning, this may run slower as it's looking for all the player\'s team. Please consider using team abbreviation in addition to player id to make the extraction faster. Defaulting to the player\'s current team in year {self.end_dt[0:4]}")
       
        
    def run_executor(self) -> pl.DataFrame:
        play_ids_df = []

        with tqdm(total=len(self.game_pks), desc="Extracting Play IDs") as progress:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = futures = {
                                    executor.submit(self._get_play_ids, game_pk, home_team, away_team): (game_pk, home_team, away_team)
                                    for game_pk, home_team, away_team in zip(self.game_pks, self.home_team, self.away_team)
                                }
                for future in concurrent.futures.as_completed(futures):
                    game_pk, _, _ = futures[future]
                    try:
                        play_ids_df.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Issue getting {game_pk}. Most Likely doesn\'t have pitch-level data. {e}")
  
                    progress.update(1)

        play_ids_df = pl.concat(play_ids_df, how = 'vertical_relaxed') # Hoping vertical relaxed fixes the Null -> Ints, Floats columns

        if play_ids_df.is_empty():
            raise pl.exceptions.NoDataError("Cannot continue, no dataframe was returned.")
        
        if self.max_return_videos:
            return play_ids_df.sample(min(self.max_return_videos, len(play_ids_df)))
        return play_ids_df
    

    def _get_play_ids(self, game_pk: int, home_team: str, away_team: str) -> (pl.DataFrame | None):
        """
        Function that extracts tha play ids for each game.

        Parameters:
            game_pk (int): The game id used to scrape the pitch-level data.
            home_team (str): The home team for the game. Used as a descriptive feature in the DataFrame.
            away_team (str): The away team for the game. Used as a descriptive feature in the DataFrame.
        
        Returns:
            DataFrame: A polars DataFrame of the pitch-level data with 40 columns.
        """

        self.rate_limiter(rate=20) # 20 Calls per second

        response = self.requests_with_retry(self.GAMEDAY_URL.format(game_pk))

        df = pl.DataFrame()
        batter_list = []
        pitcher_list = []
        p_throws_list = []
        inning_list = []
        inning_top_bot_list = []

        data = response.json()
        for play in data['allPlays']:
            batter = play['matchup']['batter']['id']
            pitcher = play['matchup']['pitcher']['id']
            p_throws = play['matchup']['pitchHand']['code']
            inning = play['about']['inning']
            inning_top_bot = play['about']['halfInning']
            for pitch in play.get('playEvents', {}):
                if not pitch:
                    self.logger.debug('Skip Pitch, no data')
                    continue

                _df = pl.json_normalize(pitch, separator='_')
                df = pl.concat([df, _df], how='diagonal')

                batter_list.append(batter) # Assures these are the same length as df
                pitcher_list.append(pitcher)
                p_throws_list.append(p_throws)
                inning_list.append(inning)
                inning_top_bot_list.append(inning_top_bot)
            
        df = df.with_columns([
            pl.Series(name="batter", values = batter_list),
            pl.Series(name="pitcher", values = pitcher_list),
            pl.Series(name="p_throws", values = p_throws_list),
            pl.Series(name="inning", values = inning_list),
            pl.Series(name="inning_top_bot", values = inning_top_bot_list),
            pl.lit(game_pk).alias("game_pk"),
            pl.lit(home_team).alias("home_team"),
            pl.lit(away_team).alias("away_team")
        ])

        # Add missing columns, select required ones, and apply filters
        for col in [c for c in self.rename_dict.keys() if c not in df.columns]:
            df = df.with_columns(pl.lit(None).alias(col))
            
        df = df.select(list(self.rename_dict.keys())).rename(self.rename_dict).filter(pl.col("play_id").is_not_null())
        
        # Apply filters on player and pitch_type
        if self.player:
            player_filter = (pl.col("batter") == self.player) | (pl.col("pitcher") == self.player)
            if self.pitch_type:
                df = df.filter(player_filter & (pl.col("pitch_type") == self.pitch_type))
            else:
                df = df.filter(player_filter)
        elif self.pitch_type:
            df = df.filter(pl.col("pitch_type") == self.pitch_type)
            
        if self.max_videos_per_game:
            return df.sample(min(self.max_videos_per_game, len(df)))
        
        return df