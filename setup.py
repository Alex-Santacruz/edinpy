from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="edinpy",
    version="0.0.1",
    description="An efficient and intuitive package for solving many-body problems using exact diagonalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Alex Santacruz",
    author_email="alex.santacruz.c@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="physics, many body, exact diagonalization",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10, <4",
    install_requires=["numpy", 'scipy'],
    project_urls={
        "Bug Reports": "https://github.com/Alex-Santacruz/edipy/issues",
        "Source": "https://github.com/Alex-Santacruz/edipy/",
    },
)