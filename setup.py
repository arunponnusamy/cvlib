from setuptools import setup

setup(name='cvlib',
      version='0.2.2',
      description='A high level, easy to use, open source computer vision library for python',
      long_description='A high level, easy to use, open source computer vision library for python',        
      url='https://github.com/arunponnusamy/cvlib.git',
      author='Arun Ponnusamy',
      author_email='hello@arunponnusamy.com',
      license='MIT',
      packages=['cvlib'],
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy', 'progressbar', 'requests', 'pillow', 'keras',
                        'imageio', 'imutils']
      )
