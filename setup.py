from setuptools import setup, find_packages

setup(
    name="SimpleNet",
    version="0.0.3",
    author="Michael Noel",
    author_email="mjn2024@gmail.com",
    description="used to build simple Neural Nets in python",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
