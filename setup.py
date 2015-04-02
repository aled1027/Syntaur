"""
Baby's first setup module.
"""

from setuptools import setup, find_packages
from os import path

def readfile(fn):
    here = path.abspath(path.dirname(__file__))
    return open('/'.join([here, fn])).read()

setup(
    name='syntaur',
    version='0.1dev'
    packages=find_packages(),
    description='A neural net library for python',
    long_description=readfile('README.rst')
    url='https://github.com/jacobmenick/Syntaur',
    author='Jacob Menick',
    author_email='jmenick@reed.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Machine Learning :: Neural Networks',
        'License :: ??????',
        'Programming Language :: Python :: 2.7',
    ],

    install_requires=dependencies,
    package_data = {'syntaur.datasets': ['patents100k.txt.gz',
                                         'englishStop.txt']},
    
    

    
)

