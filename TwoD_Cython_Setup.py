# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:07:04 2021

@author: angus
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name="TwoD_Cython", ext_modules=cythonize("TwoD_Cython.pyx"), 
      include_dirs=[np.get_include()])