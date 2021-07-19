try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
    name='resonant_lsm',
    version='0.0',
    author='Scott Sibole',
    packages=['resonant_lsm',
             ])
