import pytest
import requests
from unittest.mock import patch, Mock

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
    response.headers = {"Content-Disposition": "attachment; filename=pitcher_hitter_catcher_detector_v3.pt"}
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

def test_bdl_api_call(success_response, error_response):
    """
    Test whether the BDL API is working correctly for the BaseballCV repo.

    Args:
        success_response (Mock): Mock successful response.
        error_response (Mock): Mock error response.
    """
    with patch('requests.get') as mock_get:
        mock_get.side_effect = [success_response, error_response]

        # Test successful call to BDL API
        response = requests.get('https://balldatalab.com/api/models/phc_detector')
        assert response.status_code == 200
        assert response.content == b"mock file content"
        assert "Content-Disposition" in response.headers
        assert response.headers["Content-Disposition"].startswith("attachment")

        # Test error call to BDL API
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            response = requests.get('https://balldatalab.com/api/models/not_model')
            response.raise_for_status()

        assert exc_info.value.response.status_code == 404
        assert exc_info.value.response.json() == {"error": "File not found"}
        
        assert mock_get.call_count == 2
        mock_get.assert_any_call('https://balldatalab.com/api/models/phc_detector')
        mock_get.assert_any_call('https://balldatalab.com/api/models/not_model')
