#!/usr/bin/env python

from setuptools import setup, find_packages


def requirements():
    '''
    load requirements file as minimal packages
    '''
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(name='sslvae',
      version='1.0.0',
      description='Implementation of SSLVAE M2 as shown in the paper Semi-supervised Learning with Deep Generative Models',
      author='Xavier Sumba',
      author_email='xavier.sumba93@ucuenca.edu.ec',
      url='https://github.com/cuent/sslvae',
      install_requires=requirements(),
      packages=find_packages()
      )
