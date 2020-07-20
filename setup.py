from setuptools import find_packages, setup

with open("./README.md") as f:
    long_description = f.read()

setup(
    name="covid-scheduling",
    description="Framework for assigning people to COVID-19 testing schedules",
    author="Metric Geometry and Gerrymandering Group",
    author_email="gerrymandr@gmail.com",
    maintainer="Parker J. Rule",
    maintainer_email="parker.rule@tufts.edu",
    long_description=long_description,
    long_description_content_type="text/x-markdown",
    url="https://github.com/vrdi/geometry-of-graph-partitions",
    packages=['covid_scheduling'],
    version='0.1',
    install_requires=['numpy', 'schema', 'ortools'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ]
)
