# setup.py

from setuptools import setup, find_packages

setup(
    name="baseline-feature",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "pandas>=1.2.4",
        "scikit-learn>=0.24.2",
        "wandb>=0.12.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "opencv-python>=4.5.3",
        "PyYAML>=5.4.1",
        "tqdm>=4.62.3"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Graph-augmented Vision Transformer for chest X-ray classification"
)

