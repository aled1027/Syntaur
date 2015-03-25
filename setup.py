"""
Baby's first setup module.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open('/'.join([here, 'dependencies.txt']), 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name='syntaur',
    version='0.0.0',
    description='A neural net library for python',
    long_description='A theano-based neural network library for Python.',
    url='https://github.com/jacobmenick/Syntaur',
    author='Jacob Menick',
    author_email='jmenick@reed.edu',
    license='?????',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Neural Networks',
        'License :: ??????',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='machine learning neural networks language vision',
    packages=find_packages(),
    install_requires=dependencies,
    package_data = {'syntaur.datasets': ['patents100k.txt.gz',
                                         'englishStop.txt']},
    
    

    
)

