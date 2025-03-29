from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
import time
import random
import requests
from typing import Any, Generator, Tuple

VALID_SEASON_DATES = {
    2015: (date(2015, 4, 5), date(2015, 11, 1)),
    2016: (date(2016, 4, 3), date(2016, 11, 2)),
    2017: (date(2017, 4, 2), date(2017, 11, 1)),
    2018: (date(2018, 3, 29), date(2018, 10, 28)),
    2019: (date(2019, 3, 20), date(2019, 10, 30)),
    2020: (date(2020, 7, 23), date(2020, 10, 27)),
    2021: (date(2021, 4, 1), date(2021, 11, 2)),
    2022: (date(2022, 4, 7), date(2022, 11, 5)),
    2023: (date(2023, 3, 30), date(2023, 11, 1)),
    2024: (date(2024, 3, 28), date(2024, 10, 30)),
    2025: (date(2025, 3, 27), date(2025, 3, 27)) # Will fix this as the season progresses, doesn't have spring training for covid.
    }

class Crawler(ABC):
    """
    Abstract Class that is used for web scraping implementation. 
    """
    def __init__(self, start_dt: str, end_dt: str = None) -> None:
        super().__init__()
        self.start_dt = start_dt
        self.end_dt = end_dt

        if self.end_dt is None:
            self.end_dt = self.start_dt
        if self.end_dt < self.start_dt: # If the date order is reversed, swap
            self.end_dt, self.start_dt = self.start_dt, self.end_dt
        
        assert self.start_dt >= '2015-03-01', 'Please make queries in Statcast Era (At least 2015).'

        self.start_dt_date, self.end_dt_date = datetime.strptime(self.start_dt, "%Y-%m-%d").date(), datetime.strptime(self.end_dt, "%Y-%m-%d").date()
        self.last_called = 0

    @abstractmethod
    def run_executor(self) -> Any:
        """
        Method that uses threading on functions calls to improve the speed of the program.

        Returns:
            Any: In this case it can be a list or None.
        """
        pass

    def rate_limiter(self, rate = 10) -> None: # Random wait calls
        """
        A function that approximates wait time calls per minute so the API isn't rate limiting the program.
        It uses some noise to prevent consistent wait times.

        Parameters:
            rate (int): The number of times the function should be called per second. Default is 10.

        Returns:
            None
        """
        current_time = time.time()
        time_between_calls = 1 / rate

        time_elapsed = current_time - self.last_called

        if time_elapsed < time_between_calls:
            time_to_wait = time_between_calls - time_elapsed
            noise = random.uniform(-5, 5)
            time_to_wait += noise
            time_to_wait = max(time_to_wait, 0)
            time.sleep(time_to_wait)

        self.last_called = time.time()

    def requests_with_retry(self, url, stream = False):
        attempts = 0
        retries = 5

        while attempts < retries:
            try:
                response = requests.get(url, stream=stream, timeout=10)
                if response.status_code == 200:
                    return response
            except Exception as e:
                print("Error Downloading URL. Attempting another")
                attempts += 1
                time.sleep(2)

        
    def _date_range(self, start_dt: date, stop: date, step: int = 1) -> Generator[Tuple[date, date], Any, None]:
        """
        Iterates over the start and end date ranges using tuples with the ranges from the step. 
        Ex) 2024-02-01, 2024-02-28, with a step of 3, it will skip every 3 days such as (2024-02-01, 2024-02-03)

        Parameters:
            start_dt (date): The starting date, represented as a datetime object.
            end_dt (date): The ending date, represented as a datetime object.
            step (int): The number of days to increment by, defaults to 1 day.

        Returns:
            Generator[Tuple[datetime, Any], None, None]
        """
        low = start_dt

        while low <= stop:
            date_span = low.replace(month=3, day=15), low.replace(month=11, day=15)
            season_start, season_end = VALID_SEASON_DATES.get(low.year, date_span)
            
            if low < season_start:
                low = season_start

                print("Skipping Offseason Dates")

            elif low > season_end:
                low, _ = VALID_SEASON_DATES.get(low.year + 1, (date(month=3, day=15, year=low.year + 1), None))
                print("Skipping Offseason Dates")
            
            if low > stop:
                return

            high = min(low + timedelta(step-1), stop)

            yield low, high

            low +=timedelta(days=step)