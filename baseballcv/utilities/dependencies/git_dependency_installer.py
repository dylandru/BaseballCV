import subprocess
import sys
import logging
from pathlib import Path
import pkg_resources

logger = logging.getLogger("BaseballCV - Git Dependency Installer")

def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed without trying to import it.

    Args:
        package_name: Name of the package to check (e.g., "git+https://github.com/...")

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    try:
        if package_name.startswith('git+'):
            package_name = package_name.split('/')[-1].split('.git')[0]
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_package(package_name: str) -> bool:
    """
    Install a single package if it's not already installed.
    
    Args:
        package_name: Name of the package to install (e.g., "git+https://github.com/...")

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    if is_package_installed(package_name):
        return True
        
    logger.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def install_git_dependencies() -> bool:
    """
    Install all Git dependencies for baseballcv.

    Returns:
        bool: True if all dependencies are installed, False otherwise.
    """
    logger.info("Installing Git dependencies for baseballcv...")
    git_deps = [
        "git+https://github.com/dylandru/yolov9.git"
    ]

    success = True
    for dep in git_deps:
        if not install_package(dep):
            success = False
            logger.error(f"Failed to install {dep}")
            continue
        logger.info(f"Installed {dep}")
    
    if success:
        marker_dir = Path.home() / ".baseballcv"
        marker_dir.mkdir(exist_ok=True)
        marker_path = marker_dir / ".baseballcv_git_deps_installed"
        marker_path.write_text('installed')

    return success

def check_and_install_dependencies() -> bool:
    """
    Check if git dependencies need to be installed and install if necessary.

    Returns:
        bool: True if all dependencies are installed, False otherwise.
    """
    marker_path = Path.home() / ".baseballcv" / ".baseballcv_git_deps_installed"
    
    if not marker_path.exists():
        return install_git_dependencies()
    
    git_deps = ["git+https://github.com/dylandru/yolov9.git"]
    if all(is_package_installed(dep) for dep in git_deps):
        return True
        
    marker_path.unlink(missing_ok=True)
    return install_git_dependencies()