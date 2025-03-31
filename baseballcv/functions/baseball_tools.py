from .utils import DistanceToZone
from baseballcv.utilities import BaseballCVLogger

class BaseballTools:
    """
    Class for analyzing baseball videos with Computer Vision.
    """
    def __init__(self, device: str = 'cpu', verbose: bool = True):
        """
        Initialize the BaseballTools class.

        Args:
            device (str): Device to use for the analysis (default is 'cpu')
            verbose (bool): Whether to print verbose output (default is True)
        """
        self.device = device
        self.verbose = verbose
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)

    def distance_to_zone(self, start_date: str = "2024-05-01", end_date: str = "2024-05-01", team_abbr: str = None, 
                         pitch_type: str = None, player: int = None,
                         max_videos: int = None, max_videos_per_game: int = None, create_video: bool = True, 
                         catcher_model: str = 'phc_detector', glove_model: str = 'glove_tracking', 
                         ball_model: str = 'ball_trackingv4', zone_vertical_adjustment: float = 0.5,
                         save_csv: bool = True, csv_path: str = None) -> list:
        """
        The DistanceToZone function calculates the distance of a pitch to the strike zone in a video, as well as
        other information about the Play ID including the frame where the ball crosses, and the distance between the 
        target and the estimated strike zone.
        
        Args:
            start_date (str): Start date of the analysis
            end_date (str): End date of the analysis
            team_abbr (str): Team to analyze
            pitch_type (str): Pitch type to analyze
            player (int): Player to analyze
            max_videos (int): Maximum number of videos to analyze
            max_videos_per_game (int): Maximum number of videos per game to analyze
            create_video (bool): Whether to create a video of the analysis
            catcher_model (str): Path to the PHCDetector model, primarily used for catching (default is YOLO model 'phc_detector')
            glove_model (str): Path to the GloveTracking model, primarily used for glove detection (default is YOLO model 'glove_tracking')
            ball_model (str): Path to the BallTracking model, primarily used for ball detection (default is YOLO model 'ball_trackingv4')
            zone_vertical_adjustment (float): Factor to adjust strike zone vertically as percentage of elbow-to-hip distance.
                                             Positive values move zone toward home plate, negative away from home plate. (default is 0.5)
            save_csv (bool): Whether to save analysis results to CSV (default is True)
            csv_path (str): Custom path for CSV file (default is results/distance_to_zone_results.csv)
            
        Returns:
            list: List of results from the DistanceToZone class for each video analyzed.
        """
        dtoz = DistanceToZone(
            device=self.device, 
            verbose=self.verbose, 
            catcher_model=catcher_model, 
            glove_model=glove_model, 
            ball_model=ball_model,
            zone_vertical_adjustment=zone_vertical_adjustment,
            logger=self.logger
        )
        
        results = dtoz.analyze(
            start_date=start_date, 
            end_date=end_date, 
            team_abbr=team_abbr, 
            player=player,
            pitch_type=pitch_type, 
            max_videos=max_videos, 
            max_videos_per_game=max_videos_per_game, 
            create_video=create_video,
            save_csv=save_csv,
            csv_path=csv_path
        )
        
        return results