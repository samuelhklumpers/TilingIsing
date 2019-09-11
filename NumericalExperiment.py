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
import sys

DEFAULT_SEEDS = [2019090814]

fh = logging.FileHandler("log.txt", mode="w")
stdout_handle = logging.StreamHandler(sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(fh)
log.addHandler(stdout_handle)

class Grid:
    def __init__(self, n, T_red, seed=DEFAULT_SEEDS[0]):
        log.info(f"Generate grid {n}x{n} with seed {seed}")

        np.random.seed(seed)
        self.grid = 2 * random.randint(0, 2, size=(n, n), dtype=np.int8) - 1
        self.T_red = T_red

    def getEnergy(self):
        mult = np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) + np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)
        
        val = -np.sum(self.grid * mult)
        
        return val

    def getAverageEnergy(self):
        return self.getEnergy() / self.grid.size

    def getAverageMagnetization(self):
        return np.average(self.grid)

    def metroStep(self):
        coordRow = random.randint(0, self.grid.shape[0])
        coordCol = random.randint(0, self.grid.shape[1])
        
        energy = self.getEnergy()
        self.grid[coordRow, coordCol] *= -1
        energyDiff = self.getEnergy() - energy

        accept = True
        if energyDiff > 0.0:
            chance = np.exp(-energyDiff / self.T_red) \
                        if self.T_red != 0 else 0.0
            accept = random.random() < chance

        if not accept:
            self.grid[coordRow, coordCol] *= -1
            
    def show(self):
        plt.figure()
        self.plotGrid(axis=plt.gca())
        plt.show(block=False)
        
    def plot(self, axis, **kwargs):
        axis.imshow(self.grid, clim=(0, 1), **kwargs)

def Test1():
    gr = Grid(50, 1.0e0)
    
    for i in range(25000):
        gr.metroStep()
        
    gr.show()
    
def Test2():
    gr = Grid(200, 0.5)
    
    for i in range(2500000):
        gr.metroStep()
        
    gr.show()
    print(gr.getAverageEnergy(), gr.getAverageMagnetization())
    
    for i in range(1000000):
        gr.metroStep()
    
    gr.show()

def Exp2_6_1():
    gr = Grid(20, 5.0)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 35000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metroStep()
            numberOfAttempts += 1

        log.info(f"At attempts {numberOfAttempts}")
        gr.show()

def Exp2_6_2():
    gr = Grid(20, 10.0)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metroStep()
            numberOfAttempts += 1
            
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(gr.getAverageMagnetization()))
        log.info("Average energy: {}".format(gr.getAverageEnergy()))
        gr.show()

def Exp2_6_3():
    gr = Grid(20, 0.5)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metroStep()
            numberOfAttempts += 1
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(gr.getAverageMagnetization()))
        log.info("Average energy: {}".format(gr.getAverageEnergy()))
        gr.show()

def Exp2_6_4():
    print("  2.6.4. At reduced temperature 0.\n")

    print("At T=0 it converges to plainly +1 or -1, so it is not that interesting.\n\
    Anyway:")

    gr = Grid(20, 0.0)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metroStep()
            numberOfAttempts += 1
            
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(gr.getAverageMagnetization()))
        log.info("Average energy: {}".format(gr.getAverageEnergy()))
        gr.show()

def Exp2_6_5():
    print("  2.6.5.\n")
        
    print("Different initial configurations reach patchlike configurations, but "
          + "due to the random nature, no one is alike. Entropy decreases from "
          + "the first to the second step, but between the third and last, this "
          + "is doesn't seem to hold.")

    np.random.seed(DEFAULT_SEEDS[0])

    seeds = np.random.randint(2**30, size=(5,))
    attempts = [100, 1000, 25000, 100000]
    
    for i in range(0, len(seeds)):
        seed = seeds[i]
        gr = Grid(20, 3.0, seed=seed)
        numberOfAttempts = 0

        f, ax = plt.subplots(nrows=1, ncols=len(attempts), sharey=True)
        
        print(f"With seed {seed}")
        
        for j in range(0, len(attempts)):
            att = attempts[j]
            while numberOfAttempts < att:
                gr.metroStep()
                numberOfAttempts += 1
            #print(f"At attempts {numberOfAttempts}")
            #print("Average momentum: {}".format(GetAverageMomentum(gr)))
            #print("Average energy: {}".format(GetAverageEnergy(gr)))
            ax[j].set_title(f"At {numberOfAttempts} \nattempts")
            gr.plot(axis=ax[j])
        plt.show()
            
def GenerateSeries():
    grid = Grid(200, 0.5, DEFAULT_SEEDS[0])
    
    for i in range(100):
        with open(f"series\\series{i:03}.dat", mode="wb") as f:
            pickle.dump(grid.grid, f)
        
        for j in range(100):
            grid.metroStep()
            
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

