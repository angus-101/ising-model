# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:06:15 2021

@author: angus
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand
import matplotlib.pyplot as plt
import time
cdef extern from "limits.h":
    int RAND_MAX

cdef create_cells(int x_size, int y_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    cdef long int[:, :] cells = np.random.choice([-1, 1], size = (y_size, x_size))
    
    return cells

cdef create_image(long[:, :] cells):
    
    # Takes the final array and converts it to an image 
    
    plt.imshow(cells, aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('2D.png')
    
cdef energy_diff(long int[:, :] cells, double inverse_temp):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
    
    cdef:
        int row, column
        int y_size = cells.shape[0]
        int x_size = cells.shape[1]
        double energy
        
    for row in range(y_size):
        for column in range(x_size):
            energy = 2 * cells[row][column] * (cells[(row - 1) % y_size][column] +
                                               cells[(row + 1) % y_size][column] +
                                               cells[row][(column - 1) % x_size] +
                                               cells[row][(column + 1) % x_size])
            
            if energy < 0 or exp(-energy * inverse_temp) * RAND_MAX > rand():
                cells[row][column] *= -1
                
    return cells

def main(int array_size, float inverse_temp, int iterations):
    
    # Does any calculations
    # This is the function imported in the run file
    
    cdef:
        int count = 0
        double start_time = time.time() 
        long[:, :] cells = create_cells(array_size, array_size)
    
    while count < iterations:
        cells = energy_diff(cells, inverse_temp)
        count += 1
        
    print(time.time() - start_time)
    create_image(cells)