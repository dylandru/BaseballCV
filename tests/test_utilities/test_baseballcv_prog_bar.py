import pytest
import time
import os
from baseballcv.utilities import ProgressBar

class TestProgressBar:
    '''
    Test suite for the ProgressBar utility class. Contains various tests to ensure the ProgressBar functions
    correctly under different scenarios, including initialization, updates, description changes, closing,
    context management, and iteration.
    '''
    @pytest.fixture(autouse=True)
    def setup_tty(self, monkeypatch) -> None:
        """
        Pytest fixture to simulate a TTY environment for all tests in this class.
        Automatically runs for each test method. Mocks `os.isatty` to always return True,
        ensuring that the progress bar is enabled by default during testing, regardless
        of the actual execution environment.

        Args:
            monkeypatch: Pytest's built-in fixture for modifying classes, methods, or functions during tests.
        """
        monkeypatch.setattr(os, 'isatty', lambda x: True)
    
    def test_init_with_iterable(self) -> None:
        """
        Test ProgressBar initialization when provided with an iterable.

        Verifies that the progress bar correctly infers the total number of items
        from the length of the iterable and sets the description as provided.
        """
        test_iterable = [1, 2, 3, 4, 5]
        progress_bar = ProgressBar(iterable=test_iterable, desc="Test")
        assert progress_bar.pbar.total == 5, "Total should be inferred from iterable length"
        assert progress_bar.pbar.desc == "Test", "Description should be set correctly"
        progress_bar.close() 
        
    def test_update(self) -> None:
        """
        Test the update method of the ProgressBar. Checks if the progress bar counter increments correctly when `update` is called.
        Also verifies that the postfix dictionary is updated correctly, including the formatting of float values.
        """
        progress_bar = ProgressBar(total=10)
        initial_n = progress_bar.pbar.n
        progress_bar.update(2)
        assert progress_bar.pbar.n == initial_n + 2
        
        progress_bar.update(1, postfix={"loss": 0.12345, "accuracy": "90%"})
        postfix_str = str(progress_bar.pbar.postfix)
        assert "loss" in postfix_str
        assert "accuracy" in postfix_str
        assert "90%" in postfix_str
        assert "0.123" in postfix_str or "0.1234" in postfix_str or "0.12345" in postfix_str
        progress_bar.close()
        
    def test_set_description(self) -> None:
        """
        Test the set_description method of the ProgressBar. Verifies description text displayed by the progress bar can be 
        dynamically updated after initialization using the `set_description` method.
        """
        progress_bar = ProgressBar(total=10, desc="Initial Desc")
        original_desc = progress_bar.pbar.desc
        new_desc = "New Description"
        progress_bar.set_description(new_desc)
        assert new_desc in progress_bar.pbar.desc
        assert progress_bar.pbar.desc != original_desc
        progress_bar.close()
        
    def test_close(self, monkeypatch) -> None:
        """
        Test the close method of the ProgressBar. Verifies that when `close` is called, the total elapsed time is calculated
        and added to the progress bar's postfix display, formatted correctly.

        Args:
            monkeypatch: Pytest fixture for mocking.
        """
        time_values = [100.0, 105.0]  # Simulate 5 seconds passing
        monkeypatch.setattr(time, 'time', lambda: time_values.pop(0))
        
        progress_bar = ProgressBar(total=10) 
        progress_bar.close() 
        
        assert "Total time" in progress_bar.pbar.postfix, "'Total time' key should be in postfix after close"
        assert "5.00s" in progress_bar.pbar.postfix, "Elapsed time should be calculated and formatted correctly"
        
    def test_context_manager(self) -> None:
        """
        Test using the ProgressBar as a context manager via a 'with' statement. Ensures progress bar can be instantiated within a `with` block,
        updated within block, and closed upon exiting the block. Also checks that the bar is not disabled by default.
        """
        with ProgressBar(total=10) as progress_bar:
            assert not progress_bar.pbar.disable, "Progress bar should be enabled"
            progress_bar.update(5)
            assert progress_bar.pbar.n == 5, "Progress bar should update within the context"
            
    def test_iteration(self) -> None:
        """
        Test iterating directly over a ProgressBar instance wrapping an iterable. Verifies ProgressBar can be used in a `for` loop to iterate over
        an sequence while simultaneously displaying progress. Checks if all items from the original iterable are yielded correctly.
        """
        test_iterable = [1, 2, 3]
        result = []
        
        for item in ProgressBar(iterable=test_iterable, desc="Iterating"):
            result.append(item)
            time.sleep(0.01) 
            
        assert result == test_iterable, "Iteration should yield all items from the original iterable"
