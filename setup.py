from setuptools import setup, find_packages

setup(
    name="baseballcv",
    version="0.1.0",
    package_dir={"baseballcv": ""},
    packages=["baseballcv"] + ["baseballcv." + p for p in find_packages()],
    install_requires=open('requirements.txt').read().splitlines(),
    author="Dylan Drummey, Carlos Marcano",
    author_email="dylandrummey22@gmail.com, c.marcano@balldatalab.com",
    description="A collection of tools and models designed to aid in the use of Computer Vision in baseball.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dylandru/BaseballCV",
    python_requires=">=3.10",
)