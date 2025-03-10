from setuptools import setup, find_packages

setup(
    name='eng209',
    version='1.0.0',
    author='ENG209',
    packages=find_packages(),
    install_requires=['numpy>=2.1', 'pandas>=2.2', 'matplotlib>=3.9', 'scikit-learn>=1.5' ],
)
