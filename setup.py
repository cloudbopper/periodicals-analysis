"""Periodicals-analysis package definition and install configuration"""

from setuptools import setup, find_packages

setuptools.setup(
    name="perysis",
    version="0.1.0",
    description="Historical organic periodicals analysis",
    author="Akshay Sood",
    url="https://github.com/cloudbopper/periodicals-analysis",
    license="MIT",
    packages=find_packages()
)

