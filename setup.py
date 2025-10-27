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
    python_requires=">=3.10,<3.13",
    install_requires=[
        "tensorflow>=2.19.0,<2.20.0",
        "tensorflow-probability>=0.25.0,<0.26.0",
        "numpy>=2.0.0,<2.3.0",
        "scipy>=1.13.0,<2.0.0",
        "matplotlib>=3.8.0,<4.0.0",
        "tqdm>=4.66.0,<5.0.0",
        "PyYAML>=6.0.0,<7.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0,<9.0.0", "pytest-cov>=4.0.0,<5.0.0"],
        "colab": [
            "tensorflow==2.19.0",
            "tensorflow-probability==0.25.0",
            "numpy>=2.0.0,<2.3.0",
            "scipy>=1.13.0,<2.0.0",
            "matplotlib>=3.8.0,<4.0.0",
            "typing-extensions>=4.9.0,<5.0.0",
            "tqdm>=4.66.0,<5.0.0",
            "PyYAML>=6.0.0,<7.0.0",
            "jedi>=0.19.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
