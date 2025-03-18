import pytest
import multiprocessing as mp
from unittest.mock import Mock
import requests
from baseballcv.functions import DataTools, LoadTools, BaseballSavVideoScraper, BaseballTools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from unittest import mock

# Add the parent directory to the path to import baseballcv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing() -> None:
    """
    Ensures that the multiprocessing start method is set to 'spawn' for tests.
    
    This fixture runs automatically once per test session and configures the
    multiprocessing start method to 'spawn' which is more compatible with
    pytest and avoids potential issues with forking processes during testing.
    
    Returns:
        None: This fixture doesn't return any value.
    """
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    return None
        
@pytest.fixture
def data_tools() -> DataTools:
    """
    Provides a DataTools instance for testing.
    
    Creates and returns a DataTools object with a reduced number of workers
    to prevent excessive resource usage during testing while still allowing
    parallel processing functionality to be tested.

    Returns:
        DataTools: An instance of DataTools with max_workers set to 2.
    """
    return DataTools(max_workers=2)

@pytest.fixture
def load_tools() -> LoadTools:
    """
    Provides a LoadTools instance for testing.
    
    Creates and returns a LoadTools object that can be used in tests to load
    datasets, models, and other resources needed for testing the baseballcv
    package functionality.

    Returns:
        LoadTools: An instance of LoadTools.
    """
    return LoadTools()

@pytest.fixture
def scraper() -> BaseballSavVideoScraper:
    """
    Provides a BaseballSavVideoScraper instance for testing.
    
    Creates and returns a BaseballSavVideoScraper object that can be used
    in tests to verify the functionality of scraping baseball videos and
    related data from Baseball Savant.

    Returns:
        BaseballSavVideoScraper: An instance of BaseballSavVideoScraper.
    """
    return BaseballSavVideoScraper()

@pytest.fixture
def baseball_tools() -> BaseballTools:
    """
    Provides a BaseballTools instance for testing.
    
    Creates and returns a BaseballTools object that can be used in tests
    to verify the functionality of baseball-specific data processing and
    analysis tools provided by the baseballcv package.

    Returns:
        BaseballTools: An instance of BaseballTools.
    """
    return BaseballTools()

@pytest.fixture
def mock_responses() -> tuple:
    """
    Provides mock HTTP responses for testing network requests.
    
    Creates and returns two mock response objects:
    1. A success response (200) with mock file content and headers
    2. An error response (404) that raises an HTTPError when raise_for_status is called
    
    These mock responses can be used to test functions that make HTTP requests
    without actually connecting to external services.

    Returns:
        tuple: A tuple containing (success_response, error_response) mock objects.
    """
    success = Mock()
    success.status_code = 200
    success.content = b"mock file content"
    success.headers = {"Content-Disposition": "attachment; filename=model.pt"}
    success.raise_for_status.return_value = None  

    # Create error response
    error = Mock()
    error.status_code = 404
    error.json.return_value = {"error": "File not found"}
    http_error = requests.exceptions.HTTPError("404 Client Error: Not Found")
    http_error.response = error  
    error.raise_for_status.side_effect = http_error  

    return success, error

@pytest.fixture
def mock_model() -> Mock:
    """
    Provides a mock model for testing.
    
    Creates and returns a mock model object that can be used in tests to
    verify the functionality of model training and evaluation.
    """
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = mock.MagicMock()
            self.vision_model.encoder = mock.MagicMock()
            self.linear = nn.Linear(10, 2)
            
        def forward(self, pixel_values):
            return {"logits": torch.rand(1, 2)}
        
    return MockModel()