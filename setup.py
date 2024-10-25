from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="hip-hop-voice-sampler",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for extracting word samples from hip-hop songs using AI transcription",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hip-hop-voice-sampler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hip-hop-sampler=sampler.cli:main",
        ],
    },
)