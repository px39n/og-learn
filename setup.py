from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="og-learn",
    version="0.1.0",
    author="Zhehao Liang",
    author_email="zliang7@nd.edu",
    description="Overfit-to-Generalization Framework for Equitable Spatiotemporal Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/og-learn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    keywords=[
        "machine learning",
        "spatiotemporal",
        "overfitting",
        "generalization",
        "spatial equity",
        "environmental modeling",
    ],
)

