from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Data-Mining-Project",
    version="0.1",
    author="Chris Nikolopoulos",
    packages=find_packages(),
    install_requires = requirements,
)