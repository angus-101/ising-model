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
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
    
comm = MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()

###############################################################################

def divide(cells):
    
    # Divides a given array into its top row, bottom row, and middle rows
    
    top = cells[0,:]
    bottom = cells[-1,:]
    middle = cells[1: -1,:]
    
    return top, bottom, middle

def join(top, bottom, middle):
    
    # Joins a top row, bottom row, and middle rows into a single array
    
    cells = np.vstack((top, middle))
    cells = np.vstack((cells, bottom))
    
    return cells

def share_down(bottom, last_processor, x_size):
    
    # Sends the bottom row of an array to the processor beneath it, 
    # and receives the bottom row of the processor above it
    
    new_top = np.empty((x_size), dtype = np.int_)
    
    if taskid == 0:
        comm.Send([bottom, MPI.LONG], dest = 1, tag = 0)
        comm.Recv([new_top, MPI.LONG], source = last_processor, tag = last_processor)
        
    elif taskid == last_processor:
        comm.Recv([new_top, MPI.LONG], source = taskid - 1, tag = taskid - 1)
        comm.Send([bottom, MPI.LONG], dest = 0, tag = last_processor)
    
    else:
        comm.Send([bottom, MPI.LONG], dest = taskid + 1, tag = taskid)
        comm.Recv([new_top, MPI.LONG], source = taskid - 1, tag = taskid - 1)
        
    return new_top

def share_up(top, last_processor, x_size):
    
    # Sends the top row of an array to the processor above it,
    # and receives the top row of the processor beneath it
    
    new_bottom = np.empty((x_size), dtype = np.int_)
    
    if taskid == 0:
        comm.Send([top, MPI.LONG], dest = last_processor, tag = 0)
        comm.Recv([new_bottom, MPI.LONG], source = 1, tag = 1)
        
    elif taskid == last_processor:
        comm.Recv([new_bottom, MPI.LONG], source = 0, tag = 0)
        comm.Send([top, MPI.LONG], dest = taskid - 1, tag = last_processor)
    
    else:
        comm.Send([top, MPI.LONG], dest = taskid - 1, tag = taskid)
        comm.Recv([new_bottom, MPI.LONG], source = taskid + 1, tag = taskid + 1)
        
    return new_bottom

def create_share(x_size):
    
    # Divides the total array into smaller rows for each processor
    # Shares up and down the top and bottom rows of each smaller array
    # Joins the shared top, bottom, and middle rows
    
    y_small = int(x_size / numtasks)
    last_processor = numtasks - 1
    cells = create_cells(x_size, y_small)
    top = divide(cells)[0]
    bottom = divide(cells)[1]
    middle = divide(cells)[2]
    new_top = share_down(bottom, last_processor, x_size)
    new_bottom = share_up(top, last_processor, x_size)
    
    cells = join(new_top, new_bottom, middle)
    
    return y_small, last_processor, cells

def mpi_ising(x_size, inverse_temp, iterations):
    
    # Carries out the Ising model calculations over a given number of iterations
    # Divides, shares, and rejoins the smaller arrays
    # The odd and even rows are carried out seperately to minimise artefacts
    
    create = create_share(x_size)
    
    y_small = create[0]
    last_processor = create[1]
    cells = create[2]
        
    for count in range(iterations):
        
        cells = energy_diff(cells, inverse_temp, 1)
        middle = divide(cells)[2]
        top = divide(middle)[0]
        bottom = divide(middle)[1]
        new_top = share_down(bottom, last_processor, x_size)
        new_bottom = share_up(top, last_processor, x_size)
        cells = join(new_top, new_bottom, middle)
        cells = energy_diff(cells, inverse_temp, 0)
        middle = divide(cells)[2]
        top = divide(middle)[0]
        bottom = divide(middle)[1]
        new_top = share_down(bottom, last_processor, x_size)
        new_bottom = share_up(top, last_processor, x_size)
        cells = join(new_top, new_bottom, middle)
        
    return cells

def final_join(cells):
    
    # Master processor combines all smaller arrays
    
    x_size = cells.shape[0]
    y_small = cells.shape[1]
    
    for processor in range(1, numtasks):
        data = np.empty((x_size, y_small), dtype = np.int_)
        comm.Recv([data, MPI.LONG], source = processor, tag = processor)
        cells = np.vstack((cells, data))
        
    return cells

def mpi_main(x_size, inverse_temp, iterations):
    
    # Worker processors send their arrays to the master, 
    # and master combines them
    
    if numtasks == 1:
        cells = create_cells(x_size, x_size)
        for count in range(iterations):
            cells = energy_diff(cells, inverse_temp, 0)
            cells = energy_diff(cells, inverse_temp, 1)
        return cells
        
    cells = mpi_ising(x_size, inverse_temp, iterations)
    
    if taskid == 0:
        cells = final_join(cells)
        return cells
    else:
        comm.Send([cells, MPI.LONG], dest = 0, tag = taskid)

###############################################################################

def create_cells(x_size, y_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    cells = np.random.choice([-1, 1], size = (y_size, x_size))
    
    return cells

def create_image(cells):
    
    # Takes the final array and converts it to an image   

    plt.imshow(cells, aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('2D.png')
    
def energy_diff(cells, inverse_temp, parity):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
    # The loop includes a parity argument so that odd and even rows can be 
    # carried out seperately
    
    y_size = cells.shape[0]
    x_size = cells.shape[1]
        
    for row in range(parity, y_size - parity, 2):
        for column in range(x_size):
            energy = 2 * cells[row][column] * (cells[(row - 1) % y_size][column] +
                                               cells[(row + 1) % y_size][column] +
                                               cells[row][(column - 1) % x_size] +
                                               cells[row][(column + 1) % x_size])
            
            if energy < 0 or np.exp(-energy * inverse_temp) > np.random.rand():
                cells[row][column] *= -1
                
    return cells

###############################################################################

def vary_size(array_size, inverse_temp, iterations):
    
    # Measures the time taken for a given array to run, repeats it twice, 
    # and calculates an average and standard deviation time
    
    start_time1 = MPI.Wtime()
    cells = mpi_main(array_size, inverse_temp, iterations)
    if taskid == 0:
        time1 = MPI.Wtime() - start_time1
        
    start_time2 = MPI.Wtime()
    cells = mpi_main(array_size, inverse_temp, iterations)
    if taskid == 0:
        time2 = MPI.Wtime() - start_time2
        
    start_time3 = MPI.Wtime()
    cells = mpi_main(array_size, inverse_temp, iterations)
    if taskid == 0:
        time3 = MPI.Wtime() - start_time3
        
    if taskid == 0:
        
        av_time = (time1 + time2 + time3) / 3
        std_time = np.sqrt(((time1 - av_time) ** 2 +
                            (time2 - av_time) ** 2 +
                            (time3 - av_time) ** 2) / 3)
        
        print(array_size,"x",array_size," average time = ",av_time)
        print(array_size,"x",array_size," time std dev = ",std_time)

###############################################################################

def main(array_size, inverse_temp, iterations):
    
    # Does any calculations
    # This is the function imported in the run file

    """

    start_time = MPI.Wtime() 
    
    cells = mpi_main(array_size, inverse_temp, iterations)
    
    if taskid == 0:
        print(MPI.Wtime() - start_time)
        create_image(cells)
        
    """
    
    vary_size(50, 0.6, 50)
    vary_size(75, 0.6, 50)
    vary_size(100, 0.6, 50)
    for i in range(1, 25):
        vary_size(i * 125, 0.6, 50)
    
    #vary_size(1000, 0.6, 50)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    