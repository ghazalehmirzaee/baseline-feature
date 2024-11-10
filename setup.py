# setup.py
from setuptools import setup, find_packages

setup(
    name="baseline-feature",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.19.2",
        "pandas>=1.2.4",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "wandb>=0.12.0",
        "pyyaml>=5.4.1",
        "tqdm>=4.62.3",
        "pytorch-lightning>=1.5.0",
        "albumentations>=1.1.0"
    ],
    author="Ghazaleh Mirzaee",
    author_email="mirzaeeghazal@gmail.com",
    description="Multi-stage framework for chest X-ray classification with graph-augmented features",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ghazalehmirzaee/baseline-feature",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)




