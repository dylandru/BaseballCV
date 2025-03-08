from .utils import DistanceToZone

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

    def distance_to_zone(self, start_date: str, end_date: str, team: str = None, pitch_call: str = None,
                         max_videos: int = None, max_videos_per_game: int = None, create_video: bool = True, 
                         phc_model: str = 'phc_detector', glove_model: str = 'glove_tracking', 
                         ball_model: str = 'ball_trackingv4') -> float:
        """
        Analyze the distance of a pitch to the strike zone in a video. Based on the DistanceToZone class. 
        Current function use only supports Ultralytics YOLO models.
        
        Args:
            start_date (str): Start date of the analysis
            end_date (str): End date of the analysis
            team (str): Team to analyze
            pitch_call (str): Pitch call to analyze
            max_videos (int): Maximum number of videos to analyze
            max_videos_per_game (int): Maximum number of videos per game to analyze
            create_video (bool): Whether to create a video of the analysis
            phc_model (str): Path to the PHCDetector model (default is  YOLO model 'phc_detector')
            glove_model (str): Path to the GloveTracking model (default is YOLO model 'glove_tracking')
            ball_model (str): Path to the BallTracking model (default is YOLO model 'ball_trackingv4')
        Returns:
            results (list): List of results from the DistanceToZone class for each video analyzed.
        """
        dtoz = DistanceToZone(device=self.device, verbose=self.verbose, phc_model=phc_model, glove_model=glove_model, ball_model=ball_model)
        results = dtoz.analyze(start_date=start_date, end_date=end_date, team=team, pitch_call=pitch_call, max_videos=max_videos, 
                     max_videos_per_game=max_videos_per_game, create_video=create_video)
        
        return results