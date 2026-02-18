"""
Setup script for EchoZero application

Installation:
    pip install -e .          # Development mode (editable install)
    pip install .             # Regular install

After installation, run with:
    echozero                  # If entry point is configured
    python -m src.main        # Or directly
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).resolve().parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).resolve().parent / "requirements.txt"
install_requires = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="echozero",
    version="0.1.0",
    description="EchoZero Audio Processing Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EchoZero",
    author_email="",
    url="https://github.com/yourusername/echozero",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "echozero=main:main",
        ],
    },
    py_modules=["main"],  # Include main.py as a module
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
)

