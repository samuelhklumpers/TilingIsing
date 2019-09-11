# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:16:30 2019

@author: Vincent
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import logging
import pickle
import os

DEFAULT_SEEDS = [2019090814]

fh = logging.FileHandler("log.txt", mode="w")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(fh)

def GenerateGrid(n, seed=DEFAULT_SEEDS[0], silent=True):
    log.info(f"Begingrid {n}x{n} with seed {seed}")

    np.random.seed(seed)
    randomInts = random.randint(0, 2, size=(n, n), dtype=bool)

    return randomInts

def GetEnergy(grid):
    grid = 2 * grid - 1 
    
    mult = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
    
    val = -np.sum(grid * mult)
    
    return val

def GetAverageEnergy(grid):
    return GetEnergy(grid) / grid.size

def GetAverageMomentum(grid):
    momentum = np.average(grid)
    momentum = 2.0 * momentum - 1.0

    return momentum

def DoOneChange(grid, reducedTemperature):
    #coordArr = []
    #for coord in grid.shape:
    #    coordArr += [random.randint(0, coord)]
    coordRow = random.randint(0, grid.shape[0])
    coordCol = random.randint(0, grid.shape[1])
    #print("At ({:}, {:}): {}".format(coordRow, coordCol,
    #           grid[coordRow, coordCol]))
    energy = GetEnergy(grid)
    energyDiff = -2 * energy

    accept = True
    if energyDiff > 0.0:
        chance = np.exp(-energyDiff / reducedTemperature) \
                    if reducedTemperature != 0 else 0.0
        accept = random.random() < chance

    if accept:
        grid[coordRow, coordCol] = not grid[coordRow, coordCol]
        #plt.imshow(grid, clim=(0, 1))
        #plt.show()

def Test1():
    gr = GenerateGrid(50)
    #plt.imshow(gr)
    plt.show()
    for i in range(25000):
        DoOneChange(gr, 1.0e0)
    ShowGrid(gr)

def ShowGrid(grid):
    plt.figure()

    #ticks = np.append(np.arange(0, grid.shape[0] - 1, grid.shape[0] // 5), [grid.shape[0]])
    #print(ticks)

    #plt.yticks(ticks)
    #plt.ylim((0, grid.shape[0]))

    #plt.xticks(np.arange(0, grid.shape[0], 2))
    plt.imshow(grid, clim=(0, 1))
    plt.show(block=False)


def Exp2_6_1():
    gr = GenerateGrid(20)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 35000]
    reducedTemperature = 5.0

    for att in attempts:
        while numberOfAttempts < att:
            DoOneChange(gr, reducedTemperature)
            numberOfAttempts += 1

        log.info(f"At attempts {numberOfAttempts}")
        ShowGrid(gr)

def Exp2_6_2():
    gr = GenerateGrid(20)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]
    reducedTemperature = 10.0

    for att in attempts:
        while numberOfAttempts < att:
            DoOneChange(gr, reducedTemperature)
            numberOfAttempts += 1
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(GetAverageMomentum(gr)))
        log.info("Average energy: {}".format(GetAverageEnergy(gr)))
        ShowGrid(gr)

def Exp2_6_3():
    gr = GenerateGrid(20)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]
    reducedTemperature = 0.5

    for att in attempts:
        while numberOfAttempts < att:
            DoOneChange(gr, reducedTemperature)
            numberOfAttempts += 1
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(GetAverageMomentum(gr)))
        log.info("Average energy: {}".format(GetAverageEnergy(gr)))
        ShowGrid(gr)

def Exp2_6_4():
    print("  2.6.4. At reduced temperature 0.\n")

    print("At T=0 it converges to plainly +1 or -1, so it is not that interesting.\n\
    Anyway:")

    gr = GenerateGrid(20)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]
    reducedTemperature = 0.0

    for att in attempts:
        while numberOfAttempts < att:
            DoOneChange(gr, reducedTemperature)
            numberOfAttempts += 1
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(GetAverageMomentum(gr)))
        log.info("Average energy: {}".format(GetAverageEnergy(gr)))
        ShowGrid(gr)

def Exp2_6_5():
    print("  2.6.5. .\n")

    print("Different initial configurations reach patchy configurations, but "
          + "due to the random nature, no one is alike.")

    np.random.seed(DEFAULT_SEEDS[0])

    seeds = np.random.randint(2**30, size=(5,))
    attempts = [100, 1000, 25000]#[1e1, 1e2, 1e3, 1e4, 25000]

    for i in range(0, len(seeds)):
        seed = seeds[i]
        gr = GenerateGrid(20, seed, silent=True)
        numberOfAttempts = 0

        reducedTemperature = 3.0

        for j in range(0, len(attempts)):
            att = attempts[j]
            while numberOfAttempts < att:
                DoOneChange(gr, reducedTemperature)
                numberOfAttempts += 1
            print(f"At attempts {numberOfAttempts}")
            print("Average momentum: {}".format(GetAverageMomentum(gr)))
            print("Average energy: {}".format(GetAverageEnergy(gr)))
            ShowGrid(gr)

def GenerateSeries():
    grid = GenerateGrid(200, DEFAULT_SEEDS[0], silent=True)
    T_red = 0.5
    
    for i in range(100):
        with open(f"series\\series{i:03}.dat", mode="wb") as f:
            pickle.dump(grid, f)
        
        for j in range(100000):
            DoOneChange(grid, T_red)
            
def AnimateSeries():
    fig = plt.figure()
    
    ims = []
    i = 0
    for fn in os.listdir("series\\"):
        with open("series\\" + fn, mode="rb") as f:
            grid = pickle.load(f)
            
        ims.append([plt.imshow(grid, clim=(0, 1)), plt.text(0.9, 1.2, i)])
        i += 1
        
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=0)

    ani.save("series.gif", writer=PillowWriter(fps=10))
    