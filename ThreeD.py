# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:56:05 2020

@author: angus
"""

import matplotlib.pyplot as plt
import numpy as np
import time

def create_cells(x_size, y_size, z_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    return np.random.choice([-1, 1], size = (z_size, y_size, x_size))

def create_image(cells):
    
    # Takes the final array and converts it to an image 
    
    x_size, y_size, z_size = cells.shape
    z1 = int(z_size / 4)
    z2 = int(z_size / 2)
    z3 = int(3 * z_size / 4)
    
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
    
def energy_diff(cells, inverse_temp):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
        
    x_size = cells.shape[2]
    y_size = cells.shape[1]
    z_size = cells.shape[0]
        
    for row in range(z_size):
        for column in range(y_size):
            for aisle in range(x_size):
                energy = 2 * cells[row][column][aisle] * (cells[(row - 1) % y_size][column][aisle] +
                                                          cells[(row + 1) % y_size][column][aisle] +
                                                          cells[row][(column - 1) % x_size][aisle] +
                                                          cells[row][(column + 1) % x_size][aisle] +
                                                          cells[row][column][(aisle - 1) % z_size] +
                                                          cells[row][column][(aisle + 1) % z_size])
                
                if energy < 0 or np.exp(-energy * inverse_temp) > np.random.rand():
                    cells[row][column][aisle] *= -1
                    
    return cells

def main(array_size, inverse_temp, iterations):
    
    # Does any calculations
    
    count = 0
    start_time = time.time()
    
    cells = create_cells(array_size, array_size, array_size)
    while count < iterations:
        cells = energy_diff(cells, inverse_temp)
        count += 1
        
    print(time.time() - start_time)
    create_image(cells)
    
main(50, 0.6, 20)