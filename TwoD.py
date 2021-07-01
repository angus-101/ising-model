# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:56:05 2020

@author: angus
"""


import numpy as np
import matplotlib.pyplot as plt
import time

def create_cells(x_size, y_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    return np.random.choice([-1, 1], size = (y_size, x_size))

def create_image(cells):
    
    # Takes the final array and converts it to an image 
    
    plt.imshow(cells, aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('2D.png')
    
def energy_diff(cells, inverse_temp):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
    
    y_size = cells.shape[0]
    x_size = cells.shape[1]
        
    for row in range(y_size):
        for column in range(x_size):
            energy = 2 * cells[row][column] * (cells[(row - 1) % y_size][column] +
                                               cells[(row + 1) % y_size][column] +
                                               cells[row][(column - 1) % x_size] +
                                               cells[row][(column + 1) % x_size])
            
            if energy < 0 or np.exp(-energy * inverse_temp) > np.random.rand():
                cells[row][column] *= -1
                
    return cells

def main(array_size, inverse_temp, iterations):
    
    # Does any calculations
    
    count = 0
    start_time = time.time()
    cells = create_cells(array_size, array_size)
    
    while count < iterations:
        cells = energy_diff(cells, inverse_temp)
        count += 1
        
    print(time.time() - start_time)
    create_image(cells)
    
main(200, 0.6, 50)