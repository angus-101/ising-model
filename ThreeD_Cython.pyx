# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:56:05 2020

@author: angus
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX

cdef create_cells(int x_size, int y_size, int z_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    cdef long[:, :, :] cells = np.random.choice([-1, 1], size = (z_size, y_size, x_size))
    
    return cells

cdef create_image(long[:, :, :] cells):
    
    # Takes the final array and converts it to an image 
    
    cdef:
        int x_size = cells.shape[2]
        int y_size = cells.shape[1]
        int z_size = cells.shape[0]
        
        Py_ssize_t z1 = int(z_size / 4)
        Py_ssize_t z2 = int(z_size / 2)
        Py_ssize_t z3 = int(3 * z_size / 4)

    plt.imshow(cells[0], aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('3D1.png')
    plt.imshow(cells[z1], aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('3D2.png')
    plt.imshow(cells[z2], aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('3D3.png')
    plt.imshow(cells[z3], aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('3D4.png')
    plt.imshow(cells[z_size - 1], aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('3D5.png')
    
cdef energy_diff(long[:, :, :] cells, double inverse_temp):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
        
    cdef:
        int x_size = cells.shape[2]
        int y_size = cells.shape[1]
        int z_size = cells.shape[0]
        int row, column, aisle
        double energy
        
    for row in range(z_size):
        for column in range(y_size):
            for aisle in range(x_size):
                energy = 2 * cells[row][column][aisle] * (cells[(row - 1) % y_size][column][aisle] +
                                                          cells[(row + 1) % y_size][column][aisle] +
                                                          cells[row][(column - 1) % x_size][aisle] +
                                                          cells[row][(column + 1) % x_size][aisle] +
                                                          cells[row][column][(aisle - 1) % z_size] +
                                                          cells[row][column][(aisle + 1) % z_size])
                
                if energy < 0 or exp(-energy * inverse_temp) * RAND_MAX > rand():
                    cells[row][column][aisle] *= -1
                    
    return cells

def main(int array_size, float inverse_temp, int iterations):
    
    # Does any calculations
    # This is the function imported in the run file
    
    cdef:
        int count = 0
        double start_time = time.time()
        long[:, :, :] cells = create_cells(array_size, array_size, array_size)
        
    while count < iterations:
        cells = energy_diff(cells, inverse_temp)
        count += 1
        
    print(time.time() - start_time)
    create_image(cells)
    