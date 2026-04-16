"""
Setup script for EchoZero application

Installation:
    pip install -e .          # Development mode (editable install)
    pip install .             # Regular install

After installation, run with:
    echozero                  # Canonical EZ2 desktop shell
    echozero-foundry          # Foundry desktop UI
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
    version="2.0.0-dev",
    description="EchoZero 2 desktop audio analysis workstation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Griffin Dennen",
    author_email="griffin@813creative.net",
    packages=find_packages(include=["echozero", "echozero.*"]),
    py_modules=["run_echozero", "run_timeline_demo"],
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require={
        "core": [],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "pytest-timeout>=2.2",
            "mypy>=1.5",
            "black>=23.7",
            "isort>=5.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "echozero=run_echozero:main",
            "echozero-demo=run_timeline_demo:main",
            "echozero-foundry=echozero.foundry.ui.main_window:run_foundry_ui",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
)
