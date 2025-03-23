import subprocess
import sys
import importlib
import logging
from pathlib import Path

logger = logging.getLogger("BaseballCV - Git Dependency Installer") #needed to avoid circular imports

def check_and_install(package_name, import_name=None):
    """
    Check if a package is installed and install it if not.
    
    Args:
        package_name: Name of the package to install (e.g., "git+https://github.com/...")
        import_name: Name to use when importing the package (defaults to package_name)
    """
    if import_name is None:
        import_name = package_name.split('/')[-1].split('.git')[0]
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        logger.info(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {e}")
            return False

def install_git_dependencies() -> bool:
    """Install Git dependencies for baseballcv."""
    logger.info("Installing Git dependencies for baseballcv...")
    git_deps = [
        "git+https://github.com/Jensen-holm/statcast-era-pitches.git@1.1",
        "git+https://github.com/dylandru/yolov9.git"
    ]

    for dep in git_deps:
        try:
            check_and_install(dep)
            logger.info(f"Installed {dep}")
        except Exception as e:
            logger.error(f"Failed to install {dep}: {e}")
            return False
        
    marker_dir = Path.home() / ".baseballcv"
    marker_dir.mkdir(exist_ok=True)
    marker_path = marker_dir / ".baseballcv_git_deps_installed"
    marker_path.write_text('installed')

    return True

def check_and_install() -> bool:
    """Check if git dependencies need to be installed and install if needed."""

    marker_path = Path.home() / ".baseballcv" / ".baseballcv_git_deps_installed"
    
    if not marker_path.exists():
        return install_git_dependencies()
    
    return True