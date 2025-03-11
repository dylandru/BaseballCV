import math
from typing import Tuple

class DistanceCalculator:
    """
    Class for calculating distances between objects in baseball videos.
    
    Primarily used for measuring the distance from the ball to the strike zone.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the DistanceCalculator.
        
        Args:
            verbose (bool): Whether to print detailed progress information
        """
        self.verbose = verbose
    
    def calculate_distance_to_zone(self, ball_center: Tuple[float, float], 
                                  strike_zone: Tuple[int, int, int, int],
                                  pixels_per_foot: float) -> Tuple[float, float, str, Tuple[float, float]]:
        """
        Calculate the distance from the ball to the nearest point on the strike zone.
        Using the MLB standard 17-inch strike zone width for calibration.
        
        Args:
            ball_center (Tuple[float, float]): (x, y) coordinates of ball center
            strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
            pixels_per_foot (float): Conversion factor from pixels to feet
            
        Returns:
            Tuple[float, float, str, Tuple[float, float]]: 
                (distance in pixels, distance in inches, position description, closest point coordinates)
        """
        ball_x, ball_y = ball_center
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        
        # Define a small tolerance (1.5 inches in pixels) to avoid false positives at the boundary
        inches_per_pixel = 12 / pixels_per_foot
        tolerance_pixels = 1.5 / inches_per_pixel
        
        # Find closest point on strike zone boundary
        closest_x = max(zone_left, min(ball_x, zone_right))
        closest_y = max(zone_top, min(ball_y, zone_bottom))
        
        # Calculate distance in pixels
        dx = ball_x - closest_x
        dy = ball_y - closest_y
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # If ball is inside strike zone or within tolerance distance, distance is 0
        inside_zone = (zone_left <= ball_x <= zone_right and zone_top <= ball_y <= zone_bottom)
        
        # Even if the coordinates suggest it's inside, double-check with distance calculation
        # This will catch edge cases where the ball is very close to the boundary
        if inside_zone and distance_pixels > tolerance_pixels:
            # Double verify since we're getting conflicting results
            if self.verbose:
                print(f"Ball coordinates suggest inside zone but distance ({distance_pixels:.2f}px) > tolerance. Re-checking...")
            
            # If it's on the edge, force it to be outside
            inside_zone = False
        
        # Calculate position description
        position = ""
        if ball_y < zone_top:
            position = "High"
        elif ball_y > zone_bottom:
            position = "Low"
        
        if ball_x < zone_left:
            position += " Inside" if position else "Inside"
        elif ball_x > zone_right:
            position += " Outside" if position else "Outside"
        
        # If we're very close to being inside but distance > 0, make it explicit
        if distance_pixels <= tolerance_pixels and not inside_zone:
            position = "Borderline " + (position if position else "Edge")
        
        # If truly inside, set distance to 0 and position to "In Zone"
        if inside_zone:
            distance_pixels = 0
            position = "In Zone"
            
        # Convert to inches
        distance_inches = distance_pixels * inches_per_pixel
        
        # Return distance and the closest point coordinates for visualization
        return distance_pixels, distance_inches, position, (closest_x, closest_y)