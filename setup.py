import os

from setuptools import setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('requirements.txt') as f:
      install_requires = f.read().splitlines()

setup(
      name='cvlib',
      version='0.2.0',
      description='A high level, easy to use, open source computer vision library for python',
      url='https://github.com/arunponnusamy/cvlib.git',
      author='Arun Ponnusamy',
      author_email='hello@arunponnusamy.com',
      license='MIT',
      packages=['cvlib'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
            'gpu':  ['tensorflow-gpu'],
      },
)
