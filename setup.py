# -*- coding: utf-8 -*-
import setuptools as st
import glob

def readme():
    with open('README.rst') as f:
        return f.read()

st.setup(name='camnl',
      version='0.1',
      description='Calibration of camera nonlinearity',
      long_description = readme(),
      long_description_content_type="text/x-rst",
      url='',
      author='Christian Schrader',
      author_email='christian.schrader@ptb.de',
      license='none',
      classifiers=[
        'Development Status :: 3 - Alpha',
#       'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=st.find_packages(),
      #scripts=['bin/bla'],
      include_package_data=True, # copy docs etc.
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
      ]
      )
