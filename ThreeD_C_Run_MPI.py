# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:18:22 2021

@author: angus
"""

import sys
from ThreeD_C_MPI import main

if int(len(sys.argv)) == 4:
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
else:
    print("Correct arguments are: array size, inverse temperature, iterations")