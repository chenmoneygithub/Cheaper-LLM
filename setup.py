"""Setup script."""

import pathlib

from setuptools import find_packages
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="cheaper-llm",
    description=("Make your LLM call cheaper."),
    long_description=README,
    long_description_content_type="text/markdown",
    version="0.0.1",
    url="https://github.com/chenmoneygithub/Cheaper-LLM",
    author="chenmoney",
    author_email="qianchen94era@gmail.com",
    license="Apache License 2.0",
    # Supported Python versions
    python_requires=">=3.8",
    install_requires=[
        "absl-py",
        "numpy",
        "langchain",
        "torch",
        "transformers",
        "openai",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
)
