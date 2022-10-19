"""Setup file for Blackwater."""
import os
import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "blackwater", "VERSION.txt")
)

with open(version_path, "r") as fd:
    version = fd.read().rstrip()

setuptools.setup(
    name="blackwater",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="quantum ml qiskit error correction mitigation",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    version=version,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
