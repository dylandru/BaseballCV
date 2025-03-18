import pytest
import requests
from unittest.mock import patch

def test_bdl_api_call(mock_responses) -> None:
    """
    Tests the BDL API with both successful and error cases.
    
    Args:
        mock_responses (tuple): A tuple containing the success and error mock responses.
    """
    success_response, error_response = mock_responses
    
    with patch('requests.get') as mock_get:
        mock_get.side_effect = [success_response, error_response]

        # Test success
        response = requests.get('https://balldatalab.com/api/models/phc_detector')
        assert response.status_code == 200
        assert response.content == b"mock file content"
        assert "Content-Disposition" in response.headers
        
        # Test error
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            response = requests.get('https://balldatalab.com/api/models/not_model')
            response.raise_for_status()
        
        assert exc_info.value.response.status_code == 404
        assert exc_info.value.response.json() == {"error": "File not found"}
        
        assert mock_get.call_count == 2
        mock_get.assert_any_call('https://balldatalab.com/api/models/phc_detector')
        mock_get.assert_any_call('https://balldatalab.com/api/models/not_model')

