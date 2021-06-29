from setuptools import setup, find_packages

setup(
    name='roc_aggregator',
    version='1.0.0',
    author='Pedro Mateus',
    url='https://gitlab.com/UM-CDS/general-tools/roc-aggregator',
    description='ROC aggregator',
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
