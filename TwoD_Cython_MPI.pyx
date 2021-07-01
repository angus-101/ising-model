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
from mpi4py import MPI
    
comm = MPI.COMM_WORLD
cdef:
    int taskid = comm.Get_rank()
    int numtasks = comm.Get_size()

###############################################################################

cdef divide(long int[:, ::1] cells):
    
    # Divides a given array into its top row, bottom row, and middle rows
    
    cdef:
        long[::1] top = cells[0,:]
        long[::1] bottom = cells[-1,:]
        long int[:, ::1] middle = cells[1: -1,:]
    
    return top, bottom, middle

cdef join(long[::1] top, long[::1] bottom, long int[:, ::1] middle):
    
    # Joins a top row, bottom row, and middle rows into a single array
    
    cdef long int[:, ::1] cells
    
    cells = np.vstack((top, middle))
    cells = np.vstack((cells, bottom))
    
    return cells

cdef share_down(long[::1] bottom, int last_processor, int x_size):
    
    # Sends the bottom row of an array to the processor beneath it, 
    # and receives the bottom row of the processor above it
    
    cdef long[::1] new_top = np.empty((x_size), dtype = np.int_)
    
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

cdef share_up(long[::1] top, int last_processor, int x_size):
    
    # Sends the top row of an array to the processor above it,
    # and receives the top row of the processor beneath it
    
    cdef long[::1] new_bottom = np.empty((x_size), dtype = np.int_)
    
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

cdef create_share(int x_size):
    
    # Divides the total array into smaller rows for each processor
    # Shares up and down the top and bottom rows of each smaller array
    # Joins the shared top, bottom, and middle rows
    
    cdef:
        int y_small = int(x_size / numtasks)
        int last_processor = numtasks - 1
        long int[:, ::1] cells = create_cells(x_size, y_small)
        long[::1] top = divide(cells)[0]
        long[::1] bottom = divide(cells)[1]
        long int[:, ::1] middle = divide(cells)[2]
        long[::1] new_top = share_down(bottom, last_processor, x_size)
        long[::1] new_bottom = share_up(top, last_processor, x_size)
    
    cells = join(new_top, new_bottom, middle)
    
    return y_small, last_processor, cells

cdef mpi_ising(int x_size, float inverse_temp, int iterations):
    
    # Carries out the Ising model calculations over a given number of iterations
    # Divides, shares, and rejoins the smaller arrays
    # The odd and even rows are carried out seperately to minimise artefacts
    
    create = create_share(x_size)
    
    cdef:
        int y_small = create[0]
        int last_processor = create[1]
        long int[:, ::1] cells = create[2]
        int count
        long[::1] top, bottom, new_top, new_bottom
        long int[:, ::1] middle
        
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

cdef final_join(long int[:, ::1] cells):
    
    # Master processor combines all smaller arrays
    
    cdef:
        int processor
        int x_size = cells.shape[0]
        int y_small = cells.shape[1]
        long int[:, ::1] data
    
    for processor in range(1, numtasks):
        data = np.empty((x_size, y_small), dtype = np.int_)
        comm.Recv([data, MPI.LONG], source = processor, tag = processor)
        cells = np.vstack((cells, data))
        
    return cells

cdef mpi_main(int x_size, float inverse_temp, int iterations):
    
    # Worker processors send their arrays to the master, 
    # and master combines them
    
    cdef:
        long int[:, ::1] cells
        int count
        
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

cdef create_cells(int x_size, int y_size):
    
    # Creates array of given size filled with random distribution of
    # either +1 or -1
    
    cdef long int[:, ::1] cells = np.random.choice([-1, 1], size = (y_size, x_size))
    
    return cells

cdef create_image(long int[:, ::1] cells):
    
    # Takes the final array and converts it to an image   

    plt.imshow(cells, aspect = 'equal', origin = 'lower', cmap = 'binary')
    plt.savefig('2D.png')
    
cdef energy_diff(long int[:, ::1] cells, double inverse_temp, int parity):
    
    # Iterates over all the cells in the array, carrying out the energy
    # calculation on each one
    # The loop includes a parity argument so that odd and even rows can be 
    # carried out seperately
    
    cdef:
        int row, column
        int y_size = cells.shape[0]
        int x_size = cells.shape[1]
        double energy
        
    for row in range(parity, y_size - parity, 2):
        for column in range(x_size):
            energy = 2 * cells[row][column] * (cells[(row - 1) % y_size][column] +
                                               cells[(row + 1) % y_size][column] +
                                               cells[row][(column - 1) % x_size] +
                                               cells[row][(column + 1) % x_size])
            
            if energy < 0 or exp(-energy * inverse_temp) * RAND_MAX > rand():
                cells[row][column] *= -1
                
    return cells

###############################################################################

cdef observables(long int[:, ::1] cells):
    
    # Calculates the average energy, heat capacity, and magnetisation of
    # an array
    
    cdef:
        double average_energy, heat_capacity, magnetisation
        double energy_sum = 0
        double energy_sum_square = 0
        double spins = 0
        int row, column
        double energy
        int y_size = cells.shape[0]
        int x_size = cells.shape[1]
        
    for row in range(y_size):
        for column in range(x_size):
            energy = 2 * cells[row][column] * (cells[(row - 1) % y_size][column] +
                                               cells[(row + 1) % y_size][column] +
                                               cells[row][(column - 1) % x_size] +
                                               cells[row][(column + 1) % x_size])
            energy_sum += energy
            energy_sum_square += energy ** 2
            spins += cells[row][column]
    
    average_energy = energy_sum / (y_size * x_size)
    heat_capacity = (abs(energy_sum_square - energy_sum ** 2)) / (10 ** 10)
    magnetisation = spins / (y_size * x_size)
        
    return average_energy, heat_capacity, magnetisation

cdef vary_size(int array_size, double inverse_temp, int iterations):
    
    # Measures the time taken for a given array to run, repeats it twice, 
    # and calculates an average and standard deviation time
    
    cdef:
        long int[:, ::1] cells
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
        
        print(array_size,"x",array_size," average time = ",av_time)
        print(array_size,"x",array_size," time std dev = ",std_time)
        
cdef vary_temp_iterations(int array_size, float inverse_temp, int iterations, int temp):
    
    # Calculates the average of the energy, heat capacity, and magnetisation,
    # with varying beta and number of iterations.
    # The magentisation can be positive or negative, so it has been 
    # divided into two measurements
    
    cdef:
        long int[:, ::1] cells1, cells2, cells3, cells4, cells5, cells6, cells7, cells8, cells9, cells10
        double energy1, energy2, energy3
        double heat1, heat2, heat3
        double mag1, mag2, mag3, mag4, mag5, mag6, mag7, mag8, mag9, mag10
        double av_energy, av_heat, av_mag_pos, av_mag_neg
        double std_energy, std_heat, std_mag_pos, std_mag_neg
        list mag_pos = []
        list mag_neg = []
        double std_mag_pos_count = 0
        double std_mag_neg_count = 0
        int i, j
        
    cells1 = mpi_main(array_size, inverse_temp, iterations)
    cells2 = mpi_main(array_size, inverse_temp, iterations)
    cells3 = mpi_main(array_size, inverse_temp, iterations)
    cells4 = mpi_main(array_size, inverse_temp, iterations)
    cells5 = mpi_main(array_size, inverse_temp, iterations)
    cells6 = mpi_main(array_size, inverse_temp, iterations)
    cells7 = mpi_main(array_size, inverse_temp, iterations)
    cells8 = mpi_main(array_size, inverse_temp, iterations)
    cells9 = mpi_main(array_size, inverse_temp, iterations)
    cells10 = mpi_main(array_size, inverse_temp, iterations)
    
    if taskid == 0:
        
        energy1, heat1, mag1 = observables(cells1)
        energy2, heat2, mag2 = observables(cells2)
        energy3, heat3, mag3 = observables(cells3)
        mag4 = observables(cells4)[2]
        mag5 = observables(cells5)[2]
        mag6 = observables(cells6)[2]
        mag7 = observables(cells7)[2]
        mag8 = observables(cells8)[2]
        mag9 = observables(cells9)[2]
        mag10 = observables(cells10)[2]
        
        if mag1 > 0:
            mag_pos.append(mag1)
        else:
            mag_neg.append(mag1)
        if mag2 > 0:
            mag_pos.append(mag2)
        else:
            mag_neg.append(mag2)
        if mag3 > 0:
            mag_pos.append(mag3)
        else:
            mag_neg.append(mag3)
        if mag4 > 0:
            mag_pos.append(mag4)
        else:
            mag_neg.append(mag4)
        if mag5 > 0:
            mag_pos.append(mag5)
        else:
            mag_neg.append(mag5)
        if mag6 > 0:
            mag_pos.append(mag6)
        else:
            mag_neg.append(mag6)
        if mag7 > 0:
            mag_pos.append(mag7)
        else:
            mag_neg.append(mag7)
        if mag8 > 0:
            mag_pos.append(mag8)
        else:
            mag_neg.append(mag8)
        if mag9 > 0:
            mag_pos.append(mag9)
        else:
            mag_neg.append(mag9)
        if mag10 > 0:
            mag_pos.append(mag10)
        else:
            mag_neg.append(mag10)
    
        av_energy = (energy1 + energy2 + energy3) / 3
        av_heat = (heat1 + heat2 + heat3) / 3
        av_mag_pos = sum(mag_pos) / len(mag_pos)
        av_mag_neg = sum(mag_neg) / len(mag_neg)
        
        std_energy = np.sqrt(((energy1 - av_energy) ** 2 +
                              (energy2 - av_energy) ** 2 +
                              (energy3 - av_energy) ** 2) / 3)
        
        std_heat = np.sqrt(((heat1 - av_heat) ** 2 +
                            (heat2 - av_heat) ** 2 +
                            (heat3 - av_heat) ** 2) / 3)
        
        for i in range(len(mag_pos)):
            std_mag_pos_count += (mag_pos[i] - av_mag_pos) ** 2
        for i in range(len(mag_neg)):
            std_mag_neg_count += (mag_neg[i] - av_mag_neg) ** 2
            
        std_mag_pos = np.sqrt(std_mag_pos_count / len(mag_pos))
        std_mag_neg = np.sqrt(std_mag_neg_count / len(mag_neg))
        
        if (temp == 1):
        
            print("Size = ",array_size,", Beta = ",inverse_temp,", average energy = ",av_energy)
            print("Size = ",array_size,", Beta = ",inverse_temp,", energy std dev = ", std_energy)
        
            print("Size = ",array_size,", Beta = ",inverse_temp,", average heat capacity = ",av_heat)
            print("Size = ",array_size,", Beta = ",inverse_temp,", heat capacity std dev = ", std_heat)
        
            print("Size = ",array_size,", Beta = ",inverse_temp,", average positive magnetisation = ",av_mag_pos)
            print("Size = ",array_size,", Beta = ",inverse_temp,", positive magnetisation std dev = ", std_mag_pos)
        
            print("Size = ",array_size,", Beta = ",inverse_temp,", average negative magnetisation = ",av_mag_neg)
            print("Size = ",array_size,", Beta = ",inverse_temp,", negative magnetisation std dev = ", std_mag_neg)
            
        else:
            
            print("Beta = ",inverse_temp," iterations = ",iterations,", average energy = ",av_energy)
            print("Beta = ",inverse_temp," iterations = ",iterations,", energy std dev = ",std_energy)
        
            print("Beta = ",inverse_temp," iterations = ",iterations,", average heat capacity = ",av_heat)
            print("Beta = ",inverse_temp," iterations = ",iterations,", heat capacity std dev = ",std_heat)
            
            print("Beta = ",inverse_temp," iterations = ",iterations,", average positive magnetisation = ",av_mag_pos)
            print("Beta = ",inverse_temp," iterations = ",iterations,", positive magnetisation std dev = ", std_mag_pos)
        
            print("Beta = ",inverse_temp," iterations = ",iterations,", average negative magnetisation = ",av_mag_neg)
            print("Beta = ",inverse_temp," iterations = ",iterations,", negative magnetisation std dev = ", std_mag_neg)
        
###############################################################################
        
def main(int array_size, float inverse_temp, int iterations):
    
    # Does any calculations
    # This is the function imported in the run file
    
    cdef:
        long int[:, ::1] cells
        int i
    
    
    cells = mpi_main(array_size, inverse_temp, iterations)
    
    if taskid == 0:
    
        create_image(cells)
    
    """
    vary_size(50, 0.6, 50)
    vary_size(75, 0.6, 50)
    vary_size(100, 0.6, 50)
    for i in range(1, 25):
        vary_size(i * 125, 0.6, 50)
    """
        
###############################################################################
    
    """
    vary_temp_iterations(5000, 0.01, 50, 1)
    for i in range(1, 41):
        vary_temp_iterations(5000, 0.025 * i, 50, 1)
    

    vary_temp_iterations(1000, 0.01, 50, 1)
    for i in range(1, 41):
        vary_temp_iterations(1000, 0.025 * i, 50, 1) 

    vary_temp_iterations(1500, 0.01, 50, 1)
    for i in range(1, 41):
        vary_temp_iterations(1500, 0.025 * i, 50, 1)
    """
   
###############################################################################

    """
    vary_temp_iterations(1000, 0.2, 1, 0)
    for i in range(1, 51):
        vary_temp_iterations(1000, 0.2, i * 2, 0)
        
    vary_temp_iterations(1000, 0.5, 1, 0)
    for i in range(1, 51):
        vary_temp_iterations(1000, 0.5, i * 2, 0)
    
    vary_temp_iterations(1000, 0.8, 1, 0)
    for i in range(1, 51):
        vary_temp_iterations(1000, 0.8, i * 2, 0)
    """
    
###############################################################################

    #vary_size(3000, 0.6, 50)
    
    
    

    
    
    
    

    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    