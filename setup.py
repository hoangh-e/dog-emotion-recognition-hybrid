from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dog-emotion-ml",
    version="1.0.0",
    author="Dog Emotion Recognition Team",
    author_email="",
    description="A comprehensive machine learning package for dog emotion recognition using hybrid approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="machine learning, emotion recognition, dog behavior, computer vision, ensemble learning",
    project_urls={
        "Documentation": "https://github.com/your-repo/dog-emotion-ml",
        "Source": "https://github.com/your-repo/dog-emotion-ml",
        "Tracker": "https://github.com/your-repo/dog-emotion-ml/issues",
    },
) 