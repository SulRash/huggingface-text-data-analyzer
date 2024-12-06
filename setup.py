from setuptools import setup, find_packages
from pathlib import Path

# Read README.md for long description
readme_path = Path("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements.txt
requirements_path = Path("requirements.txt")
requirements = [
    line.strip()
    for line in requirements_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
] if requirements_path.exists() else [
    "transformers>=4.0.0",
    "datasets>=2.0.0",
    "torch>=1.0.0",
    "rich>=10.0.0",
    "spacy>=3.0.0",
    "pandas>=1.0.0",
    "numpy>=1.19.0",
    "scikit-learn>=0.24.0",
    "tqdm>=4.0.0",
]

setup(
    name="huggingface-text-data-analyzer",
    version="1.1.0",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*"]),
    
    # Dependencies
    install_requires=requirements,
    
    # CLI Entry point
    entry_points={
        "console_scripts": [
            "analyze-dataset=huggingface_text_data_analyzer.cli:main",
        ],
    },
    
    # Metadata
    author="Sultan Alrashed",
    author_email="sultan.m.rashed@gmail.com",
    description="A comprehensive tool for analyzing text datasets from HuggingFace's datasets library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SulRash/huggingface-text-data-analyzer",
    project_urls={
        "Bug Tracker": "https://github.com/SulRash/huggingface-text-data-analyzer/issues",
        "Documentation": "https://github.com/SulRash/huggingface-text-data-analyzer#readme",
        "Source Code": "https://github.com/SulRash/huggingface-text-data-analyzer",
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
        "Operating System :: OS Independent",
    ],
    
    # Additional package data
    package_data={
        "huggingface_text_data_analyzer": [
            "src/*.py",
        ],
    },
    include_package_data=True,
    
    # Keywords for PyPI
    keywords=[
        "nlp",
        "text-analysis",
        "huggingface",
        "datasets",
        "machine-learning",
        "data-analysis",
        "text-processing",
    ],
)