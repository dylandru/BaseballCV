import pytest
import subprocess
import pkg_resources
from unittest.mock import patch, MagicMock
from baseballcv.utilities.dependencies.git_dependency_installer import (
    is_package_installed,
    install_package,
    install_git_dependencies,
    check_and_install_dependencies
)

class TestGitDependencyInstaller:
    """Test cases for the git dependency installer module."""
    package_name = "git+https://github.com/dylandru/yolov9.git"
    expected_name = "yolov9"

    @pytest.fixture
    def mock_home_dir(self, tmp_path):
        """Create a temporary home directory for testing."""
        with patch('pathlib.Path.home', return_value=tmp_path):
            yield tmp_path

    @pytest.fixture
    def mock_pkg_resources(self):
        """Mock pkg_resources for testing package installation status."""
        with patch('pkg_resources.get_distribution') as mock_get_dist:
            yield mock_get_dist

    @pytest.mark.parametrize("package_name,expected_name,is_installed", [
        (package_name, expected_name, True),
        (package_name, expected_name, False),
        ("yolov9", "yolov9", True),
        ("yolov9", "yolov9", False),
    ])
    def test_is_package_installed(self, mock_pkg_resources, package_name, expected_name, is_installed):
        """Test checking if a package is installed with various inputs."""
        mock_pkg_resources.reset_mock()
        
        if is_installed:
            mock_pkg_resources.return_value = MagicMock()
        else:
            mock_pkg_resources.side_effect = pkg_resources.DistributionNotFound("Package not found")

        result = is_package_installed(package_name)
        assert result is is_installed
        mock_pkg_resources.assert_called_once_with(expected_name)

    @pytest.mark.parametrize("package_name,install_success,check_call_raises", [
        (package_name, True, None),
        (package_name, False, subprocess.CalledProcessError(1, "pip install")),
        (expected_name, True, None),
        (expected_name, False, subprocess.CalledProcessError(1, "pip install")),
    ])
    @patch('baseballcv.utilities.dependencies.git_dependency_installer.is_package_installed')
    @patch('subprocess.check_call')
    def test_install_package(self, mock_check_call, mock_is_installed, package_name, install_success, check_call_raises):
        """Test package installation with various scenarios."""
        mock_is_installed.return_value = False
        
        if check_call_raises:
            mock_check_call.side_effect = check_call_raises

        result = install_package(package_name)
        assert result is install_success
        mock_check_call.assert_called_once()
        mock_is_installed.assert_called_once()

    @pytest.mark.parametrize("install_success,should_create_marker", [
        (True, True),
        (False, False),
    ])
    @patch('baseballcv.utilities.dependencies.git_dependency_installer.install_package')
    def test_install_git_dependencies(self, mock_install_package, mock_home_dir, install_success, should_create_marker):
        """Test git dependencies installation with success and failure cases."""
        mock_install_package.return_value = install_success
        assert install_git_dependencies() is install_success
        
        marker_path = mock_home_dir / ".baseballcv" / ".baseballcv_git_deps_installed"
        assert marker_path.exists() is should_create_marker
        if should_create_marker:
            assert marker_path.read_text() == 'installed'

    @pytest.mark.parametrize("has_marker,packages_installed,should_install,should_remove_marker", [
        (True, True, False, False),    # Has marker, all packages installed -> no install needed
        (True, False, True, True),     # Has marker, missing packages -> install needed, remove marker
        (False, True, True, False),    # No marker, all packages installed -> install needed (to create marker)
        (False, False, True, False),   # No marker, missing packages -> install needed
    ])
    @patch('baseballcv.utilities.dependencies.git_dependency_installer.is_package_installed')
    @patch('baseballcv.utilities.dependencies.git_dependency_installer.install_git_dependencies')
    def test_check_and_install_dependencies(self, mock_install, mock_is_installed, mock_home_dir,
                                          has_marker, packages_installed, should_install, should_remove_marker):
        """Test dependency check with various marker and package states.
        
        Args:
            has_marker: Whether the marker file exists
            packages_installed: Whether all packages are installed
            should_install: Whether install_git_dependencies should be called
            should_remove_marker: Whether the marker file should be removed
        """
        if has_marker:
            marker_dir = mock_home_dir / ".baseballcv"
            marker_dir.mkdir(exist_ok=True)
            marker_path = marker_dir / ".baseballcv_git_deps_installed"
            marker_path.write_text('installed')

        mock_is_installed.return_value = packages_installed
        mock_install.return_value = True

        assert check_and_install_dependencies() is True
        
        assert mock_install.called is should_install
        
        marker_path = mock_home_dir / ".baseballcv" / ".baseballcv_git_deps_installed"
        assert marker_path.exists() is (has_marker and not should_remove_marker)
