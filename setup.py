from setuptools import setup, find_packages

setup(
    name='roc_aggregator',
    version='1.0.0',
    description='ROC aggregator',
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
