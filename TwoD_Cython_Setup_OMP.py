# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:07:04 2021

@author: angus
"""

from distutils.core import setup
from distutils.extension import Extension 
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "TwoD_Cython_OMP",
        ["TwoD_Cython_OMP.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp'],
    )
]

setup(name="TwoD_Cython_OMP", ext_modules=cythonize(ext_modules), 
      include_dirs=[np.get_include()])