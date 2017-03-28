#!/usr/bin/env python

from setuptools import setup

setup(name='pyhuckel',
      version='1.0',
      description='Calculate energy bands for 2D sp2 materials with the Huckel approach',
      author='Davide Olianas',
      author_email='ubuntupk@gmail.com',
      py_modules=['huckel'],
      install_requires=['numpy', 'scipy'],
      tests_require=['pycodestyle']
     )
