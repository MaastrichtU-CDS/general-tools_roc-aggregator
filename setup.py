from setuptools import setup, find_packages

setup(
    name='dROC',
    version='0.1.0',
    description='dROC',
    packages=find_packages(include=['dROC', 'dROC.*']),
    install_requires=[
        'numpy'
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-mock'
    ],
)
