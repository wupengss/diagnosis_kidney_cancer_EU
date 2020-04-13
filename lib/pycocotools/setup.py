from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='_mask',
      ext_modules=cythonize("_mask.pyx"),
      include_dirs=[np.get_include()]
      )