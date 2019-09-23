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

#logging.reset()
fh = logging.FileHandler("log.txt", mode="w")
stdout_handle = logging.StreamHandler(sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(fh)
log.addHandler(stdout_handle)

class IGrid:
    def getEnergy(self):
        ...

    def getAverageEnergy(self):
        ...

    def getAverageMagnetization(self):
        ...

    def metro(self):
        ...

    def wolff(self):
        ...

class SquareGrid(IGrid):
    def __init__(self, n, redT, seed=DEFAULT_SEEDS[0]):
        log.info(f"Generate grid {n}x{n} with seed {seed}")

        np.random.seed(seed)
        self.grid = 2 * random.randint(0, 2, size=(n, n), dtype=np.int8) - 1
        self.redT = np.float64(redT)

    def getEnergy(self):
        mult = np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) + np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)

        val = -np.sum(self.grid * mult)

        return val

    def getEnergyAt(self, i, j):
        v = self.grid[i, j]
        E = 0
        neighs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for offset in neighs:
            E += -v * self.grid.take(i + offset[0], mode="wrap", axis=0).take(j + offset[1], mode="wrap")

        return E

    def getAverageEnergy(self):
        return self.getEnergy() / self.grid.size

    def getAverageMagnetization(self):
        return np.average(self.grid)

    def metro(self):
        i = random.randint(0, self.grid.shape[0])
        j = random.randint(0, self.grid.shape[1])

        dE = -2 * self.getEnergyAt(i, j)

        accept = True
        if dE > 0.0:
            chance = np.exp(-dE / self.redT) \
                        if self.redT != 0 else 0.0
            accept = random.random() < chance

        if accept:
            self.grid[i, j] *= -1

    def wolff(self):
        prev_err = np.seterr(all='ignore')
        neighs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        i = random.randint(0, self.grid.shape[0])
        j = random.randint(0, self.grid.shape[1])

        dE = -2 * self.getEnergyAt(i, j)
        p_start = np.exp(-dE / self.redT)

        if random.rand() > p_start:
            return

        val0 = self.grid[i, j]
        beta = 1 / self.redT
        p = 1 - np.exp(-2.0 * beta)

        visited = np.full(self.grid.shape, 1, dtype=np.int8)
        q = queue.Queue()
        q.put((i, j))
        visited[(i, j)] = -1

        while not q.empty():
            l = q.get()

            for dl in neighs:
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
        self.plot(axis=plt.gca())
        plt.show(block=False)

    def plot(self, axis, **kwargs):
        axis.imshow(self.grid, clim=(0, 1), **kwargs)

class ExternalGrid:
    def __init__(self, n, t_func, t_callback=None, dE_func=None, seed=DEFAULT_SEEDS[0]):
        """
        The Grid object represents a square of n x n arrangement of particles with
        spin +1/-1.

        Args:
            Int n:  The sidelength of the grid
            Scalar|Array|Func t_func:  If Scalar, the reduced temperature of the whole grid,
                if Array, if ndim == 1 and size == 1 the reduced temperature of the whole grid,
                otherwise the reduced temperature at each cell.
                If Func, a function that takes the coordinates of a cell and returns its reduced temperature.

            Func dE_func:   A function that takes the coordinates of a cell and returns a modifier for its dE during flips.
        """

        log.info(f"Generate grid {n}x{n} with seed {seed}")

        np.random.seed(seed)
        self.grid = 2 * random.randint(0, 2, size=(n, n), dtype=np.int8) - 1

        if np.isscalar(t_func):
            self.T_red = t_func
            self.t_func = lambda i, j: self.T_red
        elif np.ndim(t_func) > 0:
            if t_func.size == 1:
                self.T_red = np.full(self.grid.shape, t_func[0])
            else:
                self.T_red = t_func

            self.t_func = lambda i, j: self.T_red[i][j]
        else:
            self.t_func = t_func

        if dE_func:
            self.dE_func = dE_func
        else:
            self.dE_func = lambda i, j: 0

        if t_callback:
            self.t_callback = t_callback
        else:
            self.t_callback = lambda i, j, dE: None

    def getEnergy(self):
        mult = np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) + np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1)

        val = -np.sum(self.grid * mult)

        return val

    def getEnergyAt(self, i, j):
        v = self.grid[i, j]
        E = 0
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for offset in neighbours:
            E += -v * self.grid.take(i + offset[0], mode="wrap", axis=0).take(j + offset[1], mode="wrap")

        return E

    def getAverageEnergy(self):
        return self.getEnergy() / self.grid.size

    def getAverageMagnetization(self):
        return np.average(self.grid)

    def metro(self):
        i = random.randint(0, self.grid.shape[0])
        j = random.randint(0, self.grid.shape[1])

        dE = -2 * self.getEnergyAt(i, j)
        dE += self.dE_func(i, j)

        T_red = self.t_func(i, j)

        accept = True
        if dE > 0.0:
            chance = np.exp(-dE / T_red) \
                        if T_red != 0 else 0.0
            accept = random.random() < chance

        if accept:
            self.t_callback(i, j, dE)
            self.grid[i, j] *= -1

    def wolff(self):
        raise NotImplementedError

    def show(self):
        plt.figure()
        self.plot(axis=plt.gca())
        plt.show(block=False)

    def plot(self, axis, **kwargs):
        axis.imshow(self.grid, clim=(0, 1), **kwargs)

# CW
class TilingConstraint:
    def __init__(self, n):
        self.n = n
        self.constraints = {}
        self.neighs = []

    def set_constraint(self, constraint, source_wind, target_wind, *windings):
        if not constraint in self.constraints:
            self.constraints[constraint] = []

        self.constraints[constraint] += [(source_wind, target_wind, windings)]

    def set_neighbours(self, constraints, repetitions=1):
        self.neighs = constraints, repetitions

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
        for k0, neighbour in enumerate(tile.neighs):
            if neighbour is None:
                continue

            for source_wind, target_wind, windings in self.constraints[neighbour.constraint]:
                k = (k0 + source_wind) % len(tile.neighs)

                if tile.neighs[k] is not None:
                    continue

                prev = tile
                curr = neighbour

                for winding in windings:
                    if curr is None:
                        break

                    i0 = curr.neighs.index(prev)
                    i = (i0 + winding) % len(curr.neighs)

                    prev, curr = curr, curr.neighs[i]

                if curr is None:
                    continue
                else:
                    i0 = curr.neighs.index(prev)
                    i = (i0 + target_wind) % len(curr.neighs)

                    tile.neighs[k] = curr
                    curr.neighs[i] = tile

        if tile.depth == 0:
            return []

        for i0, neighbour in enumerate(tile.neighs):
            if neighbour is not None:
                break

        if neighbour is None:
            i0 = 0
            j0 = 0
        else:
            j0 = self.neighs[0].index(neighbour.constraint)

        new_neighs = []

        for dx in range(len(self.neighs[0]) * self.neighs[1]):
            i = (i0 + dx) % len(tile.neighs)
            j = (j0 + dx) % len(self.neighs[0])

            if tile.neighs[i] is None:
                neigh = Tile(1, self.neighs[0][j].n)
                neigh.constraint = self.neighs[0][j]
                neigh.depth = tile.depth - 1

                tile.neighs[i] = neigh
                neigh.neighs[0] = tile

                new_neighs += [neigh]

        return new_neighs

class TileGrid(IGrid):
    def __init__(self, constraint, depth, redT):
        self.constraint = constraint
        self.depth = depth
        self.redT = redT

        self.rep = constraint.generate(depth=depth)
        self.lis = self.rep.toList()

    def getEnergy(self):
        E = sum(t.getEnergyAt() for t in self.lis)

        return E

    def getAverageEnergy(self):
        return self.getEnergy() / len(self.lis)

    def getAverageMagnetization(self):
        m = sum(t.spin for t in self.lis)

        return m / len(self.lis)

    def metro(self):
        raise NotImplementedError

    def wolff(self):
        if self.redT <= 0:
            return

        start = random.choice(self.lis)

        self.unvisit()
        dE = -2 * start.getEnergyAt()
        p0 = np.exp(-dE / self.redT)

        if random.rand() > p0:
            return

        beta = 1 / self.redT
        p = 1 - np.exp(-2 * beta)

        start._wolff(p, start.spin)
        self.unvisit()

    def unvisit(self):
        for t in self.lis:
            t.visited = False

    def corecurse(self, rep, f, default=None):
        q = queue.Queue()

        q.put(rep)
        rep.visited = True
        r = [] if default is None else default

        while not q.empty():
            t = q.get()

            for n in t.neighs:
                if n is None or n.visited:
                    continue
                n.visited = True
                q.put(n)

            r = f(t, r)

        self.unvisit()

        return r

    def display(self, fig=None, ax=None, show=True):
        if fig is None:
            fig = plt.figure()
            ax = fig.subplots()

        orientation = np.array([0, 1])    #mpl is ondersteboven, maar wij werken dubbel ondersteboven dus :/
        r0 = np.array([0, 0])             #idk
        prev = 0
        r = 1.0

        self.unvisit()
        self.rep._display(ax, r, orientation, r0, prev)
        self.unvisit()

        if show:
            fig.show()

        return fig, ax

class Tile:
    def __init__(self, spin, n_neighs):
        self.spin = random.choice([-1.0, 1.0])#spin
        self.neighs = [None] * n_neighs
        self.visited = False
        self.r = None

    def _display(self, ax, r, orientation, r0, prev):
        n = len(self.neighs)

        dr = r / (2 * np.tan(np.pi / n))

        if isinstance(prev, Tile):
            i0 = self.neighs.index(prev)

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

            if self.neighs[i] is not None:
                self.neighs[i]._display(ax, r, orientation, r0 + dr * orientation, self)

            orientation = R.dot(orientation)

    def getEnergyAt(self):
        dn = -sum(self.spin * neigh.spin for neigh in self.neighs if neigh is not None)

        return dn

    def _wolff(self, p, v0):
        if self.visited:
            return
        self.visited = True

        self.spin = -v0

        for neigh in self.neighs:
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

        for neigh in self.neighs:
            if neigh is not None and not neigh.visited:
                l += [neigh]
                neigh.visited = True

        return l