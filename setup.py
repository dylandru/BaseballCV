from setuptools import setup, find_packages

setup(
    name="baseballcv",
    version="0.1.0",
    package_dir={"baseballcv": ""},
    packages=["baseballcv"] + ["baseballcv." + p for p in find_packages()],
    install_requires=[
        'bs4==0.0.2',
        'cryptography==43.0.1',
        'opencv-python==4.10.0.84',
        'pybaseball==2.2.7',
        'pytest==8.3.2',
        'ultralytics>=8.2.90,!=8.3.41,!=8.3.42',
        'transformers==4.46.3',
        'peft==0.13.2',
        'streamlit==1.37.0',
        'streamlit-image-coordinates==0.1.6',
        'streamlit-drawable-canvas==0.9.3',
        'awscli==1.36.5',
        'pytest-cov==6.0.0',
        'supervision==0.25.0',
        'pillow==10.3.0',
        'tensorboard>=2.15.0',
        'wandb==0.19.1'
    ],
    dependency_links=[
        'git+https://github.com/Jensen-holm/statcast-era-pitches.git#egg=statcast_era_pitches'
    ],
    author="Dylan Drummey, Carlos Marcano",
    author_email="dylandrummey22@gmail.com, c.marcano@balldatalab.com",
    description="A collection of tools and models designed to aid in the use of Computer Vision in baseball.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dylandru/BaseballCV",
    python_requires=">=3.10",
    license="MIT",
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ]
)