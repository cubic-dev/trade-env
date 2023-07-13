from setuptools import setup, find_packages

setup(
    name='trading_env',
    version='1.0.0',
    author='xinjiyuan97',
    author_email='xinjiyuan97@163.com',
    description='A description of my package',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'gym'
    ],
)
