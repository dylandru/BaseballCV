"""BaseballCV - A collection of tools and models designed to aid in the use of Computer Vision in baseball."""

__version__ = "0.1.23"

try:
    from .utilities.dependencies.git_dependency_installer import check_and_install_dependencies
    check_and_install_dependencies()
except Exception as e:
    print(f"Warning: Failed to check/install git dependencies: {e}")
    print("You may need to manually install dependencies using: from baseballcv")




