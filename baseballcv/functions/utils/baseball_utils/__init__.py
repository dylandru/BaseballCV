# In baseballcv/functions/utils/baseball_utils/__init__.py
from .distance_to_zone import DistanceToZone
from .glove_tracker import GloveTracker
from .command_analyzer import CommandAnalyzer
from .event_detector import EventDetector # Make sure this line is present

__all__ = ['DistanceToZone', 'GloveTracker', 'CommandAnalyzer', 'EventDetector'] # And EventDetector is here