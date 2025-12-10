from setuptools import setup, find_packages

setup(
    name="quaternion-mamba-fusion",
    version="0.1.0",
    author="Your Name",
    description="Quaternion Mamba for Multi-Modal Image Fusion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "scikit-image>=0.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)