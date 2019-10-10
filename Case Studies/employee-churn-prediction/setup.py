import os

from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='Employee Churn Prediction',
    version='1-SNAPSHOT',
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'future'],
    description='Employee Churn Prediction Module',
    packages=find_packages(exclude='tests'),
    install_requires=[
        'h2o==3.16.0.2',
        'pandas',
        'numpy'
    ]
)
