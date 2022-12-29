import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stannum",
    version="0.9.0",
    author="Feng Liang",
    author_email="feng.liang@kaust.edu.sa",
    description="Fusing Taichi into PyTorch",
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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
