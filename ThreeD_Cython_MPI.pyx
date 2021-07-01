# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:56:05 2020

@author: angus
"""


import matplotlib.pyplot as plt
import numpy as np
import time
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX
from mpi4py import MPI
    
comm = MPI.COMM_WORLD
cdef:
    int taskid = comm.Get_rank()
    int numtasks = comm.Get_size()
    
###############################################################################

cdef divide(long int[:, :, ::1] cells):
    
    # Divides a given array into its top slice, bottom slice, and middle slices
    
    cdef:
        int z_size = cells.shape[0]
        long[:, :, ::1] top = cells[0:1, :, :]
        long[:, :, ::1] bottom = cells[(z_size - 1):z_size, :, :]
        long int[:, :, ::1] middle = cells[1: -1, :, :]
    
    return top, bottom, middle

cdef join(long[:, :, ::1] top, long[:, :, ::1] bottom, long int[:, :, ::1] middle):
    
    # Joins a top slice, bottom slice, and middle slices into a single array
    
    cdef long int[:, :, ::1] cells
    
    cells = np.vstack((top, middle))
    cells = np.vstack((cells, bottom))
    
    return cells

cdef share_down(long[:, :, ::1] bottom, int last_processor, int z_size):
    
    # Sends the bottom row of an array to the processor beneath it, 
    # and receives the bottom row of the processor above it
    
    cdef long[:, :, ::1] new_top = np.empty((1, z_size, z_size), dtype = np.int_)
    
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

cdef share_up(long[:, :, ::1] top, int last_processor, int z_size):
    
    # Sends the top row of an array to the processor above it,
    # and receives the top row of the processor beneath it
    
    cdef long[:, :, ::1] new_bottom = np.empty((1, z_size, z_size), dtype = np.int_)
    
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

cdef create_share(int z_size):
    
    # Divides the total array into smaller rows for each processor
    # Shares up and down the top and bottom rows of each smaller array
    # Joins the shared top, bottom, and middle rows
    
    cdef:
        int z_small = int(z_size / numtasks)
        int last_processor = numtasks - 1
        long int[:, :, ::1] cells = create_cells(z_small, z_size, z_size)
        long[:, :, ::1] top = divide(cells)[0]
        long[:, :, ::1] bottom = divide(cells)[1]
        long int[:, :, ::1] middle = divide(cells)[2]
        long[:, :, ::1] new_top = share_down(bottom, last_processor, z_size)
        long[:, :, ::1] new_bottom = share_up(top, last_processor, z_size)
    
    cells = join(new_top, new_bottom, middle)
    
    return z_small, last_processor, cells

cdef mpi_ising(int z_size, float inverse_temp, int iterations):
    
    # Carries out the Ising model calculations over a given number of iterations
    # Divides, shares, and rejoins the smaller arrays
    # The odd and even rows are carried out seperately to minimise artefacts
    
    create = create_share(z_size)
    
    cdef:
        int z_small = create[0]
        int last_processor = create[1]
        long int[:, :, ::1] cells = create[2]
        int count
        long[:, :, ::1] top, bottom, new_top, new_bottom
        long int[:, :, ::1] middle
        
    for count in range(iterations):
        
        cells = energy_diff(cells, inverse_temp, 1)
        middle = divide(cells)[2]
        top = divide(middle)[0]
        bottom = divide(middle)[1]
        new_top = share_down(bottom, last_processor, z_size)
        new_bottom = share_up(top, last_processor, z_size)
        cells = join(new_top, new_bottom, middle)
        cells = energy_diff(cells, inverse_temp, 0)
        middle = divide(cells)[2]
        top = divide(middle)[0]
        bottom = divide(middle)[1]
        new_top = share_down(bottom, last_processor, z_size)
        new_bottom = share_up(top, last_processor, z_size)
        cells = join(new_top, new_bottom, middle)
        
    return cells

cdef final_join(long int[:, :, ::1] cells):
    
    # Master processor combines all smaller arrays
    
    cdef:
        int processor
        int x_size = cells.shape[2]
        int y_size = cells.shape[1]
        int z_small = cells.shape[0]
        long int[:, :, ::1] data
    
    for processor in range(1, numtasks):
        data = np.empty((z_small, y_size, x_size), dtype = np.int_)
        comm.Recv([data, MPI.LONG], source = processor, tag = processor)
        cells = np.vstack((cells, data))
        
    return cells

cdef mpi_main(int z_size, float inverse_temp, int iterations):
    
    # Worker processors send their arrays to the master, 
    # and master combines them
    
    cdef:
        long int[:, :, ::1] cells
        int count
        
    if numtasks == 1:
        cells = create_cells(z_size, z_size, z_size)
        for count in range(iterations):
            cells = energy_diff(cells, inverse_temp, 0)
            cells = energy_diff(cells, inverse_temp, 1)
        return cells
        
    cells = mpi_ising(z_size, inverse_temp, iterations)
    
    if taskid == 0:
        cells = final_join(cells)
        return cells
    else:
        comm.Send([cells, MPI.LONG], dest = 0, tag = taskid)
    
###############################################################################

cdef create_cells(int z_size, int y_size, int x_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    cdef long int[:, :, ::1] cells = np.random.choice([-1, 1], size = (z_size, y_size, x_size))
    return cells

cdef create_image(long int[:, :, ::1] cells):
    
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
    
cdef energy_diff(long int[:, :, ::1] cells, double inverse_temp, int parity):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
    # The loop includes a parity argument so that odd and even rows can be 
    # carried out seperately
        
    cdef:
        int x_size = cells.shape[2]
        int y_size = cells.shape[1]
        int z_size = cells.shape[0]
        int row, column, aisle
        double energy
        
    for row in range(parity, z_size - parity, 2):
        for column in range(y_size):
            for aisle in range(x_size):
                energy = 2 * cells[row][column][aisle] * (cells[(row - 1) % z_size][column][aisle] +
                                                          cells[(row + 1) % z_size][column][aisle] +
                                                          cells[row][(column - 1) % y_size][aisle] +
                                                          cells[row][(column + 1) % y_size][aisle] +
                                                          cells[row][column][(aisle - 1) % x_size] +
                                                          cells[row][column][(aisle + 1) % x_size])
                
                if energy < 0 or exp(-energy * inverse_temp) * RAND_MAX > rand():
                    cells[row][column][aisle] *= -1
                    
    return cells

###############################################################################

cdef vary_size(int array_size, double inverse_temp, int iterations):
    
    # Measures the time taken for a given array to run, repeats it twice, 
    # and calculates an average and standard deviation time
    
    cdef:
        long int[:, :, ::1] cells
        double time1, time2, time3, av_time, std_time
    
    cdef double start_time1 = MPI.Wtime()
    cells = mpi_main(array_size, inverse_temp, iterations)
    if taskid == 0:
        time1 = MPI.Wtime() - start_time1
        
    cdef double start_time2 = MPI.Wtime()
    cells = mpi_main(array_size, inverse_temp, iterations)
    if taskid == 0:
        time2 = MPI.Wtime() - start_time2
        
    cdef double start_time3 = MPI.Wtime()
    cells = mpi_main(array_size, inverse_temp, iterations)
    if taskid == 0:
        time3 = MPI.Wtime() - start_time3
        
    if taskid == 0:
        
        av_time = (time1 + time2 + time3) / 3
        std_time = np.sqrt(((time1 - av_time) ** 2 +
                            (time2 - av_time) ** 2 +
                            (time3 - av_time) ** 2) / 3)
        
        print(array_size,"x",array_size,"x",array_size," average time = ",av_time)
        print(array_size,"x",array_size,"x",array_size," time std dev = ",std_time)

###############################################################################

def main(int array_size, float inverse_temp, int iterations):
    
    # Does any calculations
    # This is the function imported in the run file
    
    cdef:
        long int[:, :, ::1] cells
        int i
    
    """
    #cdef double start_time = MPI.Wtime() 
    
    cells = mpi_main(array_size, inverse_temp, iterations)
    
    if taskid == 0:
        #print(MPI.Wtime() - start_time)
    
        create_image(cells)
    
    
    #for i in range(2, 13):
        #vary_size(i * 25, 0.6, 50)
    """
    
    vary_size(200, 0.6, 50)
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    