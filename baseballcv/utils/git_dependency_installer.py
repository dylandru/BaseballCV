import subprocess
import sys
import os
from pathlib import Path

def install_git_dependencies() -> bool:
    """Install Git dependencies for baseballcv."""
    print("Installing Git dependencies for baseballcv...")
    git_deps = [
        "git+https://github.com/Jensen-holm/statcast-era-pitches.git@1.1",
        "git+https://github.com/dylandru/yolov9.git"
    ]

    for dep in git_deps:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dep}: {e}")
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