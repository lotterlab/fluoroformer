from setuptools import setup, find_packages
import os

# Read requirements.txt
with open("requirements.txt") as f:
    install_requires = f.read().strip().splitlines()

setup(
    name="fluoroformer",
    version="0.1.0",
    author="Marc Harary",
    author_email="marc@ds.dfci.harvard.edu",
    description="Attention-based multiple instance learning (ABMIL) for multiplexed immunofluorsence (mIF) images.",
    long_description=open("README.md").read(),  # Ensure a README.md is present
    long_description_content_type="text/markdown",
    url="https://github.com/lotterlab/fluoroformer",  # Replace with actual repository URL
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust license if needed
        "Operating System :: OS Independent",
    ],
)
