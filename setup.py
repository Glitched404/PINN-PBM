"""
Setup configuration for PINN-PBM package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pinn-pbm",
    version="0.1.0",
    author="Research Team",
    description="Physics-Informed Neural Networks for Population Balance Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.0.0",
        "tensorflow-probability>=0.20.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
