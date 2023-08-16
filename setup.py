from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="scqpth",
    version="0.0.1",
    author="Andrew Butler",
    author_email="",
    description="Strictly convex quadratic programming torch solver",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ipo-lab/",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
)
