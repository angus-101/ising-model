# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:17:45 2021

@author: angus
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name="ThreeD_Cython_OMP", ext_modules=cythonize("ThreeD_Cython_OMP.pyx"), include_dirs=[np.get_include()])