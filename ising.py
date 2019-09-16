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
