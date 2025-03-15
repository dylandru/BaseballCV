import pytest
import multiprocessing as mp
from unittest.mock import Mock
import requests
from baseballcv.functions import DataTools, LoadTools, BaseballSavVideoScraper, BaseballTools
from baseballcv.model import YOLOv9, PaliGemma2, Florence2, DETR
import os
import sys
from unittest import mock

# Add the parent directory to the path to import baseballcv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session", autouse=True)
def setup_multiprocessing() -> None:
    """
    Ensures that the multiprocessing start method is set to 'spawn' for tests.
    """
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    return None
        
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
def baseball_tools() -> BaseballTools:
    """
    Provides a BaseballTools instance for testing.

    Returns:
        BaseballTools: An instance of BaseballTools.
    """
    return BaseballTools()

@pytest.fixture
def yolo_model():
    """
    Fixture to provide the YOLOv9 class for testing.
    This avoids importing the actual class in every test function.
    """
    from baseballcv.model import YOLOv9
    return YOLOv9

@pytest.fixture
def paligemma_model():
    """
    Fixture to provide the PaliGemma2 class for testing.
    This avoids importing the actual class in every test function.
    """
    from baseballcv.model import PaliGemma2
    return PaliGemma2

@pytest.fixture
def florence_model():
    """
    Fixture to provide the Florence2 class for testing.
    This avoids importing the actual class in every test function.
    """
    from baseballcv.model.vlm.florence2.florence2 import Florence2
    return Florence2

@pytest.fixture
def detr_model() -> DETR:
    """
    Provides a DETR instance for testing.

    Returns:
        DETR: An instance of DETR.
    """
    return DETR()

@pytest.fixture
def mock_responses() -> tuple:
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
def mock_torch_cuda():
    """
    Fixture to mock torch.cuda for CPU-only testing environments.
    """
    with mock.patch('torch.cuda.is_available', return_value=False), \
         mock.patch('torch.cuda.empty_cache', return_value=None), \
         mock.patch('torch.cuda.memory_allocated', return_value=0):
        yield

@pytest.fixture
def mock_torch_mps():
    """
    Fixture to mock torch.backends.mps for CPU-only testing environments.
    """
    with mock.patch('torch.backends.mps.is_available', return_value=False):
        yield

@pytest.fixture
def cpu_only_env(mock_torch_cuda, mock_torch_mps):
    """
    Fixture to ensure tests run in a CPU-only environment.
    """
    yield

@pytest.fixture
def temp_dir():
    """
    Fixture to provide a temporary directory for test files.
    """
    import tempfile
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def dummy_image():
    """
    Fixture to provide a dummy image for testing.
    """
    from PIL import Image
    image = Image.new('RGB', (224, 224), color='white')
    temp_path = "temp_test_image.jpg"
    image.save(temp_path)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

