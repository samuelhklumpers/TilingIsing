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
import queue

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
        self.T_red = np.float64(T_red)

    def getEnergy(self):
        mult = np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) + np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)
        
        val = -np.sum(self.grid * mult)
        
        return val
        
    def getFlipDiff(self, i, j):
        v = self.grid[i, j]
        E = 0
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for offset in neighbours:
            E += -v * self.grid.take(i + offset[0], mode="wrap", axis=0).take(j + offset[1], mode="wrap")
            
        return -2 * E

    def getAverageEnergy(self):
        return self.getEnergy() / self.grid.size

    def getAverageMagnetization(self):
        return np.average(self.grid)

    def metroStep(self):
        coordRow = random.randint(0, self.grid.shape[0])
        coordCol = random.randint(0, self.grid.shape[1])
        
        energyDiff = self.getFlipDiff(coordRow, coordCol)

        accept = True
        if energyDiff > 0.0:
            chance = np.exp(-energyDiff / self.T_red) \
                        if self.T_red != 0 else 0.0
            accept = random.random() < chance

        if accept:
            self.grid[coordRow, coordCol] *= -1

    def wolffStep(self):
        prev_err = np.seterr(all='ignore')
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        i = random.randint(0, self.grid.shape[0])
        j = random.randint(0, self.grid.shape[1])

        dE = self.getFlipDiff(i, j)
        p_start = np.exp(-dE / self.T_red)

        if random.rand() > p_start:
            return

        val0 = self.grid[i, j]
        beta = 1 / self.T_red
        p = 1 - np.exp(-beta)

        visited = np.full(self.grid.shape, 1, dtype=np.int8)
        q = queue.Queue()
        q.put((i, j))
        visited[(i, j)] = -1

        while not q.empty():
            l = q.get()

            for dl in neighbours:
                l2 = (l[0] + dl[0], l[1] + dl[1])
                l2 = (l2[0] % self.grid.shape[0], l2[1] % self.grid.shape[1])
                
                if visited[l2] < 0:
                    continue

                if self.grid[l] != self.grid[l2]:
                    continue
                
                if random.rand() < p:
                    visited[l2] = -1
                    q.put(l2)

        self.grid *= visited
        
        np.seterr(**prev_err)
            
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
    grid = Grid(50, 0.5, DEFAULT_SEEDS[0])
    
    for i in range(100):
        with open(f"series\\series{i:03}.dat", mode="wb") as f:
            pickle.dump(grid.grid, f)
    
        for j in range(100000):
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

def CreateSeries():
    grid = Grid(60, 10, DEFAULT_SEEDS[0])

    ims = []
    fig = plt.figure()
    
    for i in range(100):
        for j in range(100000):
            grid.metroStep()

        ims.append([plt.imshow(grid.grid, clim=(0, 1)), plt.text(0.9, 1.2, i)])
        grid.T_red -= 9.5 / 100
        
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=0)

    ani.save("series.gif", writer=PillowWriter(fps=10))

def PhaseTransition():
    import scipy.ndimage.filters as filters
    
    grid = Grid(30, 10, DEFAULT_SEEDS[0])

    T = []
    E = []
    m = []

    for i in range(100):
        grid.T_red = 10 - i / 10
        T += [grid.T_red]
        
        for j in range(10000):
            grid.metroStep()

        E += [grid.getAverageEnergy()]
        m += [grid.getAverageMagnetization()]

    plt.figure()
    plt.plot(T, filters.gaussian_filter1d(E, 5), label="E")
    plt.legend()
    plt.figure()
    plt.plot(T[1:], filters.gaussian_filter1d(np.diff(E), 5), label="C")
    plt.legend()
    plt.figure()
    plt.plot(T, m, label="m")
    plt.legend()
    plt.show(block=False)

def CreateSeriesWolff():
    grid = Grid(1000, 1.0, DEFAULT_SEEDS[0])

    ims = []
    fig = plt.figure()
    
    for i in range(100):
        for j in range(10):
            grid.wolffStep()

        ims.append([plt.imshow(grid.grid, clim=(0, 1)), plt.text(0.9, 1.2, i)])
        
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=0)

    ani.save("series.gif", writer=PillowWriter(fps=10))
