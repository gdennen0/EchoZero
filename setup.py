"""Compatibility shim for setuptools-based tooling.

Package metadata and dependency declarations live in ``pyproject.toml``.
This file remains only so tools that still expect ``setup.py`` continue to work.
"""

from setuptools import setup


setup()
