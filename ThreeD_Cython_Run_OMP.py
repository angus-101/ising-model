# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:18:22 2021

@author: angus
"""

import sys
from ThreeD_Cython_OMP import main

if int(len(sys.argv)) == 5:
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
else:
    print("Usage: {} <ITERATIONS>".format(sys.argv[0]))