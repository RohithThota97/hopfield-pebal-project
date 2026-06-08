from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hopfield-ood-segmentation",
    version="0.1.0",
    author="Rohith Thota",
    author_email="your.email@example.com",
    description="Modern Hopfield Networks for Out-of-Distribution Detection in Semantic Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RohithThota97/Modern-Hopfield-Networks-for-Out-of-Distribution-Detection-in-Semantic-Segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "scikit-image>=0.18.0",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "tensorboard>=2.5.0",
        "wandb>=0.10.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.3.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.1.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "black>=21.0", "flake8>=3.9.0"],
    },
)
