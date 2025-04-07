import os
from tqdm import tqdm
import time
from typing import Any, Iterator

class ProgressBar:
    """
    A custom progress bar for BaseballCV operations with enhanced display features.
    
    Provides visual feedback for various operations like model training, dataset generation,
    and annotation processes. This class wraps tqdm with baseball-themed styling and additional
    convenience methods for tracking progress in machine learning and computer vision tasks.
    """
    def __init__(self, iterable=None, total: int = None, desc: str = "Processing", unit: str = "it", 
                 color: str = "green", disable: bool = False, bar_format: str = None, postfix: dict = None, initial: int = 0):
        """
        Initialize the progress bar with BaseballCV styling.
        
        Args:
            iterable: Optional iterable to wrap with the progress bar. If provided, total is inferred.
            total: Total number of items to process. Required if iterable is not provided.
            desc: Description text to display before the progress bar.
            unit: Unit of items being processed (e.g., 'frames', 'images', 'batches').
            color: Color of the progress bar (supports tqdm color options).
            disable: Whether to disable the progress bar display completely.
            bar_format: Custom format string for the progress bar. If None, uses BaseballCV default.
            postfix: Dictionary of key-value pairs to display at the end of the bar.
            initial: Initial counter value for the progress bar.
            
        Returns:
            None
        """
        self.start_time = time.time()
        
        baseballcv_format = "BaseballCV ⚾ | {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}{postfix}"
        
        self.pbar = tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            colour=color,
            dynamic_ncols=True,
            leave=True,
            disable = disable or not os.isatty(1),  # Disable in non-interactive environments
            bar_format=bar_format or baseballcv_format,
            ascii=" ▏▎▍▌▋▊▉█",
            postfix=postfix,
            initial=initial
        )
        
    def update(self, n: int = 1, postfix: dict = None) -> None:
        """
        Update the progress bar by incrementing the counter and optionally updating the postfix.
        
        Args:
            n: Number of items to increment the counter by.
            postfix: Dictionary of key-value pairs to display at the end of the bar.
                     Float values are automatically formatted to 4 decimal places.
        
        Returns:
            None
        """
        if postfix:
            formatted_postfix = {}
            for key, value in postfix.items():
                if isinstance(value, float):
                    formatted_postfix[key] = f"{value:.4f}"
                else:
                    formatted_postfix[key] = value
            self.pbar.set_postfix(formatted_postfix)
        self.pbar.update(n)
    
    def set_description(self, desc: str) -> None:
        """
        Update the description text of the progress bar.
        
        Args:
            desc: New description text to display before the progress bar.
        
        Returns:
            None
        """
        self.pbar.set_description(f"{desc}")
    
    def close(self) -> None:
        """
        Close the progress bar and display the total elapsed time.
        
        This method finalizes the progress bar, showing the total time taken since
        initialization and releasing any resources.
        
        Returns:
            None
        """
        elapsed = time.time() - self.start_time
        self.pbar.set_postfix({"Total time": f"{elapsed:.2f}s"})
        self.pbar.close()
        
    def __enter__(self) -> 'ProgressBar':
        """
        Enable context manager functionality with 'with' statements.
        
        Returns:
            self: The ProgressBar instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Handle context manager exit by closing the progress bar.
        
        Args:
            exc_type: Exception type if an exception was raised in the context.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
            
        Returns:
            None
        """
        self.close()
        
    def __iter__(self) -> Iterator[Any]:
        """
        Allow iteration over the wrapped iterable with progress tracking.
        
        Returns:
            iterator: An iterator over the items in the wrapped iterable.
        """
        return iter(self.pbar)
