# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:16:30 2019

@author: Vincent
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import logging
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

        return -4 * E

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
        p = 1 - np.exp(-2 * beta)

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

                if val0 != self.grid[l2]:
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



# CW
class TilingConstraint:
    def __init__(self, n):
        self.n = n
        self.constraints = {}
        self.neighbours = []

    def set_constraint(self, constraint, source_wind, target_wind, *windings):
        if not constraint in self.constraints:
            self.constraints[constraint] = []

        self.constraints[constraint] += [(source_wind, target_wind, windings)]

    def set_neighbours(self, constraints, repetitions=1):
        self.neighbours = constraints, repetitions

    def generate(self, depth):
        q = queue.Queue()
        t0 = Tile(0, self.n)
        t0.depth = depth
        t0.constraint = self
        q.put(t0)

        while not q.empty():
            t = q.get()
            for n in t.constraint._generate(t):
                q.put(n)

        return t0

    def _generate(self, tile):
        for k0, neighbour in enumerate(tile.neighbours):
            if neighbour is None:
                continue

            for source_wind, target_wind, windings in self.constraints[neighbour.constraint]:
                k = (k0 + source_wind) % len(tile.neighbours)

                if tile.neighbours[k] is not None:
                    continue

                prev = tile
                curr = neighbour

                for winding in windings:
                    if curr is None:
                        break

                    i0 = curr.neighbours.index(prev)
                    i = (i0 + winding) % len(curr.neighbours)

                    prev, curr = curr, curr.neighbours[i]

                if curr is None:
                    continue
                else:
                    i0 = curr.neighbours.index(prev)
                    i = (i0 + target_wind) % len(curr.neighbours)

                    tile.neighbours[k] = curr
                    curr.neighbours[i] = tile

        if tile.depth == 0:
            return []

        for i0, neighbour in enumerate(tile.neighbours):
            if neighbour is not None:
                break

        if neighbour is None:
            i0 = 0
            j0 = 0
        else:
            j0 = self.neighbours[0].index(neighbour.constraint)

        new_neighs = []

        for dx in range(len(self.neighbours[0]) * self.neighbours[1]):
            i = (i0 + dx) % len(tile.neighbours)
            j = (j0 + dx) % len(self.neighbours[0])

            if tile.neighbours[i] is None:
                neigh = Tile(1, self.neighbours[0][j].n)
                neigh.constraint = self.neighbours[0][j]
                neigh.depth = tile.depth - 1

                tile.neighbours[i] = neigh
                neigh.neighbours[0] = tile

                new_neighs += [neigh]

        return new_neighs

class Tile:
    def __init__(self, spin, n_neighbours):
        self.spin = 1#random.choice([-1.0, 1.0])#spin
        self.neighbours = [None] * n_neighbours
        self.visited = False
        self.r = None

    def corecurse(self, l, f, default=None):
        q = queue.Queue()

        q.put(self)
        self.visited = True
        r = [] if default is None else default

        while not q.empty():
            t = q.get()

            for n in t.neighbours:
                if n is None or n.visited:
                    continue
                n.visited = True
                q.put(n)

            r = f(t, r)

        self.unvisit(l)

        return r

    def display(self, l, fig=None, ax=None, show=True):
        if fig is None:
            fig = plt.figure()
            ax = fig.subplots()

        orientation = np.array([0, 1])    #mpl is ondersteboven, maar wij werken dubbel ondersteboven dus :/
        r0 = np.array([0, 0])             #idk
        prev = 0
        r = 1.0

        self.unvisit(l)
        self._display(ax, r, orientation, r0, prev)
        self.unvisit(l)

        if show:
            fig.show()

        return fig, ax

    def _display(self, ax, r, orientation, r0, prev):
        n = len(self.neighbours)

        dr = r / (2 * np.tan(np.pi / n))

        if isinstance(prev, Tile):
            i0 = self.neighbours.index(prev)

            if self.r is None:
                r0 = r0 + dr * orientation

                self.r = r0

            ax.plot([self.r[0], prev.r[0]], [self.r[1], prev.r[1]], 'k-', zorder=3)
        else:
            i0 = prev
            self.r = r0

        if self.visited:
            return
        self.visited = True

        mfc = 'b' if self.spin < 0 else 'r'
        ax.scatter(self.r[0], self.r[1], s=400, c=mfc, marker='o', alpha=1, zorder=4)

        orientation = -orientation

        c, s = np.cos(2 * np.pi / n), np.sin(2 * np.pi / n)
        R = np.array([[c, s], [-s, c]])

        for di in range(n):
            i = (i0 + di) % n

            if self.neighbours[i] is not None:
                self.neighbours[i]._display(ax, r, orientation, r0 + dr * orientation, self)

            orientation = R.dot(orientation)

    def getEnergy(self):
        dn = -sum(self.spin * neigh.spin for neigh in self.neighbours if neigh is not None)

        return dn

    def wolff(self, T_red, l):
        if T_red <= 0:
            return

        self.unvisit(l)
        dE = -2 * self.getEnergy()
        p0 = np.exp(-dE / T_red)

        if random.rand() > p0:
            return
        
        beta = 1 / T_red
        p = 1 - np.exp(-2 * beta)

        self._wolff(p, self.spin)
        self.unvisit(l)

    def _wolff(self, p, v0):
        if self.visited:
            return
        self.visited = True

        self.spin = -v0

        for neigh in self.neighbours:
            if neigh is None:
                continue
            
            if neigh.spin == v0 and random.rand() < p:
                neigh._wolff(p, v0)

    def toList(self):
        l = []
        q = queue.Queue()

        q.put(self)
        self.visited = True

        while not q.empty():
            t = q.get()
            l += [t]
            
            for n in t._toList():
                q.put(n)
                
        for t in l:
            t.visited = False
            
        return l

    def _toList(self):
        l = []

        for neigh in self.neighbours:
            if neigh is not None and not neigh.visited:
                l += [neigh]
                neigh.visited = True

        return l

    @classmethod
    def unvisit(self, l):
        for t in l:
            t.visited = False
