# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:07:43 2021

@author: angus
"""

import sys
from TwoD_Cython_OMP import main

if int(len(sys.argv)) == 5:
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
else:
    print("Correct arguments are: array size, inverse temperature, iterations, threads")