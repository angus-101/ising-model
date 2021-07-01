# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:56:05 2020

@author: angus
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand
import matplotlib.pyplot as plt
import time
cdef extern from "limits.h":
    int RAND_MAX
from cython.parallel cimport prange
cimport openmp

###############################################################################

cdef create_cells(int x_size, int y_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    cdef long int[:, ::1] cells = np.random.choice([-1, 1], size = (y_size, x_size))
    
    return cells

cdef create_image(long int[:, ::1] cells):
    
    # Takes the final array and converts it to an image   

    plt.imshow(cells, aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('2D.png')
    
cdef energy_diff(long int[:, ::1] cells, double inverse_temp, int threads):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
    
    cdef:
        int row, column
        int y_size = cells.shape[0]
        int x_size = cells.shape[1]
        double energy
        
    for row in prange(y_size, nogil=True, num_threads=threads):
        for column in range(x_size):
            energy = 2 * cells[row][column] * (cells[(row - 1) % y_size][column] +
                                               cells[(row + 1) % y_size][column] +
                                               cells[row][(column - 1) % x_size] +
                                               cells[row][(column + 1) % x_size])
            
            if energy < 0 or exp(-energy * inverse_temp) * RAND_MAX > rand():
                cells[row][column] *= -1
                
    return cells

###############################################################################

cdef vary_size(int array_size, double inverse_temp, int iterations, int threads):
    
    # Measures the time taken for a given array to run, repeats it twice, 
    # and calculates an average and standard deviation time
    
    cdef:
        long int[:, ::1] cells
        double time1, time2, time3, av_time, std_time
        int i
    
    cdef double start_time1 = openmp.omp_get_wtime()
    cells = create_cells(array_size, array_size)
    for i in range(iterations):
        cells = energy_diff(cells, inverse_temp, threads)
    time1 = openmp.omp_get_wtime() - start_time1
        
    cdef double start_time2 = openmp.omp_get_wtime()
    cells = create_cells(array_size, array_size)
    for i in range(iterations):
        cells = energy_diff(cells, inverse_temp, threads)
    time2 = openmp.omp_get_wtime() - start_time2
        
    cdef double start_time3 = openmp.omp_get_wtime()
    cells = create_cells(array_size, array_size)
    for i in range(iterations):
        cells = energy_diff(cells, inverse_temp, threads)
    time3 = openmp.omp_get_wtime() - start_time3
        
    av_time = (time1 + time2 + time3) / 3
    std_time = np.sqrt(((time1 - av_time) ** 2 +
                        (time2 - av_time) ** 2 +
                        (time3 - av_time) ** 2) / 3)
        
    print(av_time, std_time)
        
###############################################################################

def main(int array_size, float inverse_temp, int iterations, int threads):
    
    # Does any calculations
    # This is the function imported in the run file
    
    cdef int i
    
    
    """
    cdef:
        double start_time = openmp.omp_get_wtime()
        
    cdef:
        long int[:, ::1] cells = create_cells(array_size, array_size)
        int count = 0
        
    for count in range(iterations):
        cells = energy_diff(cells, inverse_temp, threads)
    
    print(openmp.omp_get_wtime() - start_time)
    create_image(cells)
    """
    
    vary_size(50, 0.6, 50, 4)
    vary_size(75, 0.6, 50, 4)
    vary_size(100, 0.6, 50, 4)
    for i in range(1, 25):
        vary_size(i * 125, 0.6, 50, 4)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    