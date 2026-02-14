"""Setup script for RLM package.

For development installation:
    pip install -e .

For production installation:
    pip install .
"""

from setuptools import setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rlm",
    version="0.1.0",
    description="Recursive Language Model - Python 3.13+ implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ondrej Krajicek",
    url="https://github.com/ondrasek/spike-claude-code-rlm",
    packages=["rlm"],
    python_requires=">=3.13",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
