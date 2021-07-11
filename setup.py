import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stannum",
    version="0.1.1",
    author="Feng Liang",
    author_email="feng.liang@kaust.edu.sa",
    description="PyTorch wrapper for Taichi data-oriented class",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ifsheldon/stannum",
    project_urls={
        "Bug Tracker": "https://github.com/ifsheldon/stannum/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "stannum"},
    packages=setuptools.find_packages(where="stannum"),
    python_requires=">=3.6",
)
