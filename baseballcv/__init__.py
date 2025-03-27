"""BaseballCV - A collection of tools and models designed to aid in the use of Computer Vision in baseball."""

__version__ = "0.1.21"

try:
    from .utilities import check_and_install
    check_and_install()
except Exception as e:
    print(f"Error checking and installing git dependencies: {e}")


