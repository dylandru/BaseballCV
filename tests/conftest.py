from unittest.mock import Mock
import pytest
import requests
from scripts import DataTools, LoadTools, BaseballSavVideoScraper

@pytest.fixture
def data_tools() -> DataTools:
    """
    Provides a DataTools instance for testing.

    Returns:
        DataTools: An instance of DataTools with max_workers set to 2.
    """
    return DataTools(max_workers=2)

@pytest.fixture
def load_tools() -> LoadTools:
    """
    Provides a LoadTools instance for testing.

    Returns:
        LoadTools: An instance of LoadTools.
    """
    return LoadTools()

@pytest.fixture
def scraper() -> BaseballSavVideoScraper:
    """
    Provides a BaseballSavVideoScraper instance for testing.

    Returns:
        BaseballSavVideoScraper: An instance of BaseballSavVideoScraper.
    """
    return BaseballSavVideoScraper()

@pytest.fixture
def success_response() -> Mock:
    """
    Fixture returning mock success response for file download.

    Returns:
        Mock: Mock response object with 200 code and content for file download.
    """
    response = Mock()
    response.status_code = 200
    response.content = b"mock file content"
    response.headers = {"Content-Disposition": "attachment; filename=pitcher_hitter_catcher_detector_v4.pt"}
    return response

@pytest.fixture
def error_response() -> Mock:
    """
    Fixture returning mock error response.

    Returns:
        Mock: Mock response object with 404 code and "file not found" error message.
    """
    response = Mock()
    response.status_code = 404
    response.json.return_value = {"error": "File not found"}
    response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
    return response 