"""
Setup configuration for Loud & Clear voice isolation system
"""

from setuptools import setup, find_packages

with open("readme.txt", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="loud-and-clear",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular voice isolation system using deep learning",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/yourusername/loud-and-clear",
    packages=find_packages(exclude=["tests", "docs", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "loudclear=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "models": ["pretrained/.gitkeep"],
        "data": ["temp/.gitkeep", "mixed/.gitkeep"],
    },
)