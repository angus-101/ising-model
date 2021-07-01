# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:07:04 2021

@author: angus
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name="TwoD_C_MPI", ext_modules=cythonize("TwoD_C_MPI.pyx"), 
      include_dirs=[np.get_include()])