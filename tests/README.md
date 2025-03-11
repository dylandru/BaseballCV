# BaseballCV Test Suite

This directory contains automated tests for the BaseballCV package. The tests ensure that all components of the package work correctly and help prevent failure upon a release or PR merge.

## Test Structure

The test suite is organized as follows:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **API Tests**: Test external API interactions

## Running Tests

To run the entire test suite:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_load_tools.py
```

To run tests with coverage reporting:

```bash
pytest --cov=baseballcv tests/ --cov-report=xml
```

## Test Files

| File | Description |
|------|-------------|
| `test_imports.py` | Verifies all package modules can be imported correctly |
| `test_load_tools.py` | Tests model and dataset loading functionality |
| `test_baseball_tools.py` | Tests baseball-specific analysis tools |
| `test_baseballcv_api.py` | Tests API interactions with BallDataLab |
| `test_dataset_tools.py` | Tests dataset generation and annotation |
| `test_savant_scraper.py` | Tests Baseball Savant video scraping |

## Test Fixtures

The `conftest.py` file defines fixtures that are used across multiple tests:

- `setup_multiprocessing`: Configures multiprocessing for tests
- `data_tools`: Provides a DataTools instance
- `load_tools`: Provides a LoadTools instance
- `scraper`: Provides a BaseballSavVideoScraper instance
- `baseball_tools`: Provides a BaseballTools instance
- `mock_responses`: Creates mock HTTP responses for API testing

## CI Integration

Tests are automatically run on GitHub Actions when:
- Code is pushed to the main branch
- A pull request is opened against the main branch

The workflow configuration is defined in `.github/workflows/pytest.yml` and `.github/workflows/build.yml`.

## Writing New Tests

When adding new functionality to BaseballCV, please add corresponding tests. Follow these guidelines:

1. Create a new test file or add to an existing one based on the component being tested
2. Use fixtures from `conftest.py` when possible
3. Keep tests small and focused on a single functionality
4. Use descriptive test names that explain what is being tested
5. Include both positive tests (expected behavior) and negative tests (error handling)

Example test structure:

```python
# test_feature_name.py
def test_feature_name(fixture1, fixture2):
    """
    Tests that [feature] behaves as expected when [condition].
    """
    # Setup
    expected_result = ...
    
    # Execute
    actual_result = fixture1.feature_name(...)
    
    # Assert
    assert actual_result == expected_result
```

```python
# conftest.py
@pytest.fixture
def fixture_name():
    """
    Fixture for [feature] functionality.
    """
    return ...
```

## Test Data

Tests use minimal data samples to keep execution fast:
- Limited date ranges (usually a single day)
- Small video samples (1-2 videos)
- Low frame counts (10 frames)

This ensures tests run quickly while still verifying functionality.
