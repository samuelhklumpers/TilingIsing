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
            
    def show(self):
        plt.figure()
        self.plotGrid(axis=plt.gca())
        plt.show(block=False)
        
    def plot(self, axis, **kwargs):
        axis.imshow(self.grid, clim=(0, 1), **kwargs)

# CW
class TilingConstraint:
    def __init__(self, n):
        self.n = n
        self.constraints = {}
        self.neighbours = []

    def set_constraint(self, constraint, source_wind, target_wind, *windings):
        self.constraints[constraint] = source_wind, target_wind, windings

    def set_neighbours(self, constraints, repetitions=1):
        self.neighbours = constraints, repetitions

    def generate(self, tile=None, depth=1):
        #import pdb
        #pdb.set_trace()
        
        if tile is None:
            tile = Tile(0, self.n)
            tile.constraint = self

        for k0, neighbour in enumerate(tile.neighbours):
            if neighbour is None:
                continue

            prev = tile
            curr = neighbour
            source_wind, target_wind, windings = self.constraints[neighbour.constraint]
            
            for winding in windings:
                if curr is None:
                    break
                
                i0 = curr.neighbours.index(prev)
                i = (i0 + winding) % len(curr.neighbours)

                prev, curr = curr, curr.neighbours[i]

            if curr is None:
                continue
            else:
                k = (k0 + source_wind) % len(self.neighbours)
                i0 = curr.neighbours.index(prev)
                i = (i0 + target_wind) % len(curr.neighbours)

                if tile.neighbours[k] is None:
                    tile.neighbours[k] = curr
                    curr.neighbours[i] = tile
                    
        if depth == 0:
            return tile

        for i0, neighbour in enumerate(tile.neighbours):
            if neighbour is not None:
                break

        if neighbour is None:
            i0 = 0
            j0 = 0
        else:
            j0 = self.neighbours[0].index(neighbour.constraint)

        for dx in range(len(self.neighbours[0]) * self.neighbours[1]):
            i = (i0 + dx) % len(tile.neighbours)
            j = (j0 + dx) % len(self.neighbours[0])

            if tile.neighbours[i] is None:
                neigh = Tile(0, self.neighbours[0][j].n)
                neigh.constraint = self.neighbours[0][j]

                tile.neighbours[i] = neigh
                neigh.neighbours[0] = tile

        for neigh in tile.neighbours:
            neigh.constraint.generate(tile=neigh, depth=depth - 1)
            
        return tile

class Tile:
    def __init__(self, spin, n_neighbours):
        self.spin = spin
        self.neighbours = [None] * n_neighbours

    def add(self, neighbour):
        self.neighbours += [neighbour]

    def set(self, neighbours):
        self.neighbours = neighbours

def CreateHexGrid():
    hex_constr = TilingConstraint(6)
    hex_constr.set_neighbours([hex_constr], 6)
    hex_constr.set_constraint(hex_constr, 1, -1, -1)
    hex_constr.set_constraint(hex_constr, -1, 1, 1)

    tile = hex_constr.generate(depth=1)

    return tile

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


#   0 
#      2
#   1
#      3

internal = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
direction1 = [(1, 0), (3, 0), (3, 2)]
direction2 = [(2, 0), (3, 0), (3, 1)]
