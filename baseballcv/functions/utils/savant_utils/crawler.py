import os
import time
import random
import requests
from functools import wraps
from datetime import datetime, date, timedelta
from typing import Any, Generator, Tuple, Union, Callable, TypeVar
from baseballcv.utilities import BaseballCVLogger

logger = BaseballCVLogger.get_logger(os.path.basename(__file__))

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
            2025: (date(2025, 3, 27), date(2025, datetime.today().month, datetime.today().day))
    }

F = TypeVar('F', bound=Callable[..., object]) # Function call type

def sanitize_date_range(start_dt: str, end_dt: str) -> Tuple[date, date]:
    """
    Sanitizes the date range from str to a date object.

    Args:
        start_dt (str): The ideal starting date, though handled if it's greater
        end_dt (str): The ideal ending date, though handled if it's less than

    Returns:
        Tuple[date, date]: The start and end date objects.
    """
    if end_dt is None:
        end_dt = start_dt

    if end_dt < start_dt:
        end_dt, start_dt = start_dt, end_dt

    if start_dt <= '2015-03-01':
        raise ValueError('Please make queries in Statcast Era (At least 2015).')

    start_dt_date, end_dt_date = datetime.strptime(start_dt, "%Y-%m-%d").date(), datetime.strptime(end_dt, "%Y-%m-%d").date()

    return start_dt_date, end_dt_date

def generate_date_range(start_dt: date, stop: date, step: int = 1) -> Generator[Tuple[date, date], Any, None]:
    """
    Function that iterates over the start and end date ranges using tuples with the ranges from the step. 
    Ex) 2024-02-01, 2024-02-28, with a step of 3, it will skip every 3 days such as (2024-02-01, 2024-02-03)

    Args:
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

        elif low > season_end:
            low, _ = VALID_SEASON_DATES.get(low.year + 1, (date(month=3, day=15, year=low.year + 1), None))
        
        if low > stop:
            return

        high = min(low + timedelta(step-1), stop)

        yield low, high

        low +=timedelta(days=step)

def requests_with_retry(url: str, stream: bool = False, session: requests.Session = None) -> (requests.Response | None):
    """
    Function that retries a request on a url if it fails. It re-attempts up to 5
    times with a 10 second timeout if it takes a while to load the page. If the request is
    re-atempted, it waits for 5 seconds before making another request.

    Args:
        url (str): The url to make the request on.
        stream (bool): If it's a video stream, it's set to True. Default to False.
        session (requests.Session, optional): An existing session to make the request on,
            preserving cookies and headers across calls. Defaults to None, which uses a
            bare `requests.get`.

    Returns:
        Response: A response to the request if successful, else None.

    Raises:
        Exception: Any error that could cause an issue with making the request. Main
        targeted error is rate limits.

    """
    attempts = 0
    retries = 5

    while attempts < retries:
        try:
            requester = session.get if session is not None else requests.get
            response = requester(url, stream=stream, timeout=10)
            if response.status_code == 200:
                return response
        except Exception as e:
            logger.warning(f"Error Downloading URL {url}.\nAttempting another: {e}\n")
            attempts += 1
            time.sleep(5)

def rate_limiter(arg: Union[F, int]) -> Union[F, Callable[[F], F]]:
    """
    A tool that pauses a function throughout it's call if it is making calls to
    the internet. It is treated as 
    1. A function that takes an integer input, the rate in seconds for which the function should
    ideally reach per second. rate=10 is ~10 function calls per second.
    2. A function call itself that uses the default rate, which is 10.

    **The goal of this decorator is to implement random wait calls for each function call.**

    Example Use:
    ```python
    @rate_limiter  # ~10 calls per second by default
    def example(): ...

    @rate_limiter(4) # ~4 calls per second
    def example(): ...
    ```

    Args:
        arg (Union[F, int]): The input for this decorator. It is made optional.

    Returns:
        Union[F, Callable[[F], F]]: The handled input function
    """
    def decorator(func: F, rate: int = 10) -> F:
        time_between_calls = 1 / rate
        last_called = 0 

        @wraps(func)
        def wrap(*args, **kwargs) -> object:
            nonlocal last_called # tell python that last_called falls within the scope of wrapper
            current_time = time.time()
            elapsed = current_time - last_called

            if elapsed < time_between_calls:
                wait_time = time_between_calls - elapsed
                noise = random.uniform(-1, 1)  # small noise in seconds
                wait_time += noise
                wait_time = max(wait_time, 0)
                time.sleep(wait_time)

            last_called = time.time()
            return func(*args, **kwargs)

        return wrap
    
    if callable(arg):
        # @rate_limiter
        return decorator(arg)
    else:
        # @rate_limiter(5)
        def wrap(func: F) -> F:
            return decorator(func, rate=arg)
        return wrap