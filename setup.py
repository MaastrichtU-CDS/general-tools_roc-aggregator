from os import path
from setuptools import setup, find_packages

current_directory = path.abspath(path.dirname(__file__))
with open(path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='roc_aggregator',
    version='1.1.2',
    author='Pedro Mateus',
    url='https://gitlab.com/UM-CDS/general-tools/roc-aggregator',
    description='ROC aggregator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include=['roc_aggregator', 'roc_aggregator.*']),
    install_requires=[
        'numpy >= 1.17'
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-mock'
    ],
)
