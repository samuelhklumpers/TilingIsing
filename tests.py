from ising import SquareGrid, DEFAULT_SEEDS, TilingConstraint, log, TileGrid, ExternalGrid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import pickle
import os
from numpy import random
from uncertainties import ufloat
import json

def Create666(depth, seed=None):
    hex_constr = TilingConstraint(6)
    hex_constr.set_neighbours([hex_constr], 6)
    hex_constr.set_constraint(hex_constr, 1, -1, -1)
    hex_constr.set_constraint(hex_constr, -1, 1, 1)

    tg = TileGrid(hex_constr, depth, 1.0, seed=seed, createID="Create666")

    return tg

def Create3636(depth, seed=None):
    tri_constr = TilingConstraint(3)
    hex_constr = TilingConstraint(6)

    tri_constr.set_neighbours([hex_constr], 3)
    hex_constr.set_neighbours([tri_constr], 6)

    tri_constr.set_constraint(hex_constr, 1, -1, -1, -1)
    tri_constr.set_constraint(hex_constr, -1, 1, 1, 1)
    hex_constr.set_constraint(tri_constr, 1, -1, -1, -1)
    hex_constr.set_constraint(tri_constr, -1, 1, 1, 1)

    tile = TileGrid(hex_constr, depth, 1.0, seed=seed, createID="Create3636")

    return tile

def Create333333(depth, seed=None):
    tri_constr = TilingConstraint(3)

    tri_constr.set_neighbours([tri_constr], 3)

    tri_constr.set_constraint(tri_constr, 1, -1, -1, -1, -1, -1)
    tri_constr.set_constraint(tri_constr, -1, 1, 1, 1, 1, 1)

    tile = TileGrid(tri_constr, depth, 1.0, seed=seed, createID="Create333333")

    return tile

def Create555(depth, seed=None):
    tri_constr = TilingConstraint(5)

    tri_constr.set_neighbours([tri_constr], 5)

    tri_constr.set_constraint(tri_constr, 1, -1, -1)
    tri_constr.set_constraint(tri_constr, -1, 1, 1)

    tile = TileGrid(tri_constr, depth, 1.0, seed=seed, createID="Create555")

    return tile

def Create33333(depth, seed=None):
    tri_constr = TilingConstraint(3)

    tri_constr.set_neighbours([tri_constr], 3)

    tri_constr.set_constraint(tri_constr, 1, -1, -1, -1, -1)
    tri_constr.set_constraint(tri_constr, -1, 1, 1, 1, 1)

    tile = TileGrid(tri_constr, depth, 1.0, seed=seed, createID="Create33333")

    return tile

def Create4444(depth, seed=None):
    sq_constr = TilingConstraint(4)

    sq_constr.set_neighbours([sq_constr], 4)

    sq_constr.set_constraint(sq_constr, 1, -1, -1, -1)
    sq_constr.set_constraint(sq_constr, -1, 1, 1, 1)

    tile = TileGrid(sq_constr, depth, 1.0, seed=seed, createID="Create4444")

    return tile

def Exp2_6_1():
    gr = SquareGrid(20, 5.0)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 35000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metro()
            numberOfAttempts += 1

        log.info(f"At attempts {numberOfAttempts}")
        gr.show()

def Exp2_6_2():
    gr = SquareGrid(20, 10.0)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metro()
            numberOfAttempts += 1

        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(gr.getAverageMagnetization()))
        log.info("Average energy: {}".format(gr.getAverageEnergy()))
        gr.show()

def Exp2_6_3():
    gr = SquareGrid(20, 0.5)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metro()
            numberOfAttempts += 1
        log.info(f"At attempts {numberOfAttempts}")
        log.info("Average momentum: {}".format(gr.getAverageMagnetization()))
        log.info("Average energy: {}".format(gr.getAverageEnergy()))
        gr.show()

def Exp2_6_4():
    print("  2.6.4. At reduced temperature 0.\n")

    print("At T=0 it converges to plainly +1 or -1, so it is not that interesting.\n\
    Anyway:")

    gr = SquareGrid(20, 0.0)
    numberOfAttempts = 0
    attempts = [1e1, 1e2, 1e3, 1e4, 25000]

    for att in attempts:
        while numberOfAttempts < att:
            gr.metro()
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
        gr = SquareGrid(20, 3.0, seed=seed)
        numberOfAttempts = 0

        f, ax = plt.subplots(nrows=1, ncols=len(attempts), sharey=True)

        print(f"With seed {seed}")

        for j in range(0, len(attempts)):
            att = attempts[j]
            while numberOfAttempts < att:
                gr.metro()
                numberOfAttempts += 1
            #print(f"At attempts {numberOfAttempts}")
            #print("Average momentum: {}".format(GetAverageMomentum(gr)))
            #print("Average energy: {}".format(GetAverageEnergy(gr)))
            ax[j].set_title(f"At {numberOfAttempts} \nattempts")
            gr.plot(axis=ax[j])
        plt.show()

def GenerateSeries():
    grid = SquareGrid(50, 0.5, DEFAULT_SEEDS[0])

    for i in range(100):
        with open(f"series\\series{i:03}.dat", mode="wb") as f:
            pickle.dump(grid.grid, f)

        for j in range(100000):
            grid.metro()

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



def ShowAnimate(gridsize=300, redT = 5.0,
                      frames=100, framechanges=300):
    seed = random.seed(DEFAULT_SEEDS[0])
    for i in range(20):
        seed = random.randint(0, 1e9)

    fig, ax = plt.subplots(figsize=(15, 15))

    grid = SquareGrid(gridsize, redT, seed)
    im = ax.imshow(grid.grid, clim=(0, 1))

    def update(frame):
        for j in range(framechanges):
            grid.wolff()
        #im = ax.imshow(grid.grid, clim=(0, 1))
        if True or (frame % 50) == 0:
            frame = frame
        im.set_data(grid.grid)
        #plt.text(0.2, 0.2, frame)
        return (im,)

    ani = animation.FuncAnimation(fig, update)#, frames=range(1000))
    plt.show()
    return ani

def CreateSeries():
    grid = SquareGrid(30, 10, DEFAULT_SEEDS[0])

    ims = []
    fig = plt.figure()

    for i in range(100):
        for _ in range(1000):
            grid.metro()

        ims.append([plt.imshow(grid.grid, clim=(0, 1)), plt.text(0.9, 1.2, i)])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=0)

    ani.save("series.gif", writer=PillowWriter(fps=10))

def FindCriticalTemp(grid, T=np.linspace(1, 10), settle_factor=15,
                    E_samples=50, sample_step_factor=2.0, show=False,
                    write_to_file=True, trial_seed=None, trial_index=None,
                    trial_length=None):
    E = []
    m = []
    C = []

    for temp in T:
        grid.redT = temp
        #log.info(f"Calculating at temperature {temp:.3f}")

        remaining = settle_factor * grid.getSize()
        while remaining > 0:
            remaining -= grid.wolff()

        E2 = []

        for _ in range(E_samples):
            remaining = sample_step_factor * grid.getSize()
            while remaining > 0:
                remaining -= grid.wolff()
            
            E2 += [grid.getAverageEnergy()]

        E += [np.mean(E2)]
        m += [grid.getAverageMagnetization()]

        C += [np.var(E2) / temp ** 2]

    if show:
        plt.figure()
        plt.plot(T, E, label="E")
        plt.legend()

    ##    plt.figure()
    ##    plt.plot(T[1:], filters.gaussian_filter1d(np.diff(E), 5), label="C")
    ##    plt.plot(T[1:], np.diff(filters.gaussian_filter1d(E, 5)), label="CAlt")
    ##    plt.legend()
    ##    plt.show(block=False)

        plt.figure()
        plt.plot(T, C, label="CFromVar")
        plt.legend()
        plt.show(block=False)

        plt.figure()
        plt.plot(T, m, label="m")
        plt.legend()
        plt.show(block=False)
    critTempNaive = T[np.argmax(C)]

    if write_to_file:
        genRepr = grid.getGenRepr()
        i = 0
        filenameFormat = f"PhaseTransitionData/{genRepr}_{{0}}.txt"
        filename = filenameFormat.format(i)
        while os.path.exists(filename):
            i += 1
            filename = filenameFormat.format(i)

        file = open(filename, "w")
        paramDict = {"genRepr": genRepr, "settle_factor": settle_factor,
                   "E_samples": E_samples, "sample_step_factor": sample_step_factor,
                   "trial_seed": trial_seed,
                   "trial_index": trial_index,
                   "trial_length": trial_length,
                   "crit_temp_naive": critTempNaive}
#        if trial_seed is not None:
#            paramDict["trial_seed"] = trial_seed
#        if trial_index is not None:
#            paramDict["trial_index"] = trial_index
#        if trial_length is not None:
#            paramDict["trial_length"] = trial_length

        file.write("# PARAM_DICT {}\n".format(json.dumps(paramDict)))
        file.write(f"# CRIT_TEMP_NAIVE {critTempNaive:.4e}\n")

        file.write(f"# FORMAT T, E, C, m\n")
        dat = np.transpose(np.array([T, E, C, m]))
        for row in dat:
            file.write(",\t".join(["{:.4f}".format(a) if 0.01 <= abs(a) <= 1000
                                   else "{:.4e}".format(a) for a in row]) + "\n")
        file.close()

    return critTempNaive, C

def TrialCriticalTemp(new_grid, trial_length=20, T=np.linspace(1, 10), settle_factor=15,
                        E_samples=100, sample_step_factor=2.0, trial_seed=None):
    if trial_seed is None:
        gen = random.RandomState()
    else:
        gen = random.RandomState(trial_seed)

    seeds = gen.randint(0, 1e8, trial_length)

    T_crits = [FindCriticalTemp(new_grid(seed=seeds[i]), T=T, settle_factor=settle_factor,
                                E_samples=E_samples, sample_step_factor=sample_step_factor,
                                show=False, trial_seed=trial_seed, trial_index=i,
                                trial_length=trial_length, write_to_file=False)[0]
            for i in range(trial_length)]

    return ufloat(np.mean(T_crits), np.std(T_crits) / np.sqrt(len(T_crits)))

def CreateSeriesWolff(seriesname="series.gif", gridsize=100, redT=1.0,
                      frames=100, framechanges=100):
    grid = SquareGrid(gridsize, redT, DEFAULT_SEEDS[0])

    ims = []
    fig = plt.figure(figsize=(15, 15))

    for i in range(frames):
        for j in range(framechanges):
            grid.wolff()

        ims.append([plt.imshow(grid.grid, clim=(0, 1)), plt.text(0.9, 1.2, i)])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=0)

    ani.save(seriesname, writer=PillowWriter(fps=10))

def UntilEquilibrium(n=100, redT=1.0, sample_time=10, epoch_time=100):
    grid = SquareGrid(n, redT)

    prev_top = 4
    prev_bot = -4

    plt.figure()

    E_axis = []

    while True:
        top = -4
        bot = 4

        E_axis += [grid.getAverageEnergy()]

        for i in range(sample_time):
            E = grid.getAverageEnergy()

            top = max(E, top)
            bot = min(E, bot)

            grid.wolff()

        if top > prev_top and bot < prev_bot:
            for j in range(1):
                for i in range(epoch_time):
                    grid.wolff()

                E_axis += [grid.getAverageEnergy()]

            plt.plot(list(range(len(E_axis))), E_axis)
            plt.show(block=True)

            return grid

        prev_top, prev_bot = top, bot

        for i in range(epoch_time):
            grid.wolff()

def HexWolff(depth=4, redT=4.0):
    tileGrid = Create666(depth)

    fig, ax = plt.subplots(figsize=(15, 15))


    for i in range(10):
        tileGrid.display()
        tileGrid.wolff()

def AnimateTile(tileGrid, redT=8.0, frameskip=1):
    fig, ax = plt.subplots(figsize=(15, 15))

    tileGrid.redT = redT

    def update(frame):
        ax.cla()

        for j in range(frameskip):
            tileGrid.wolff()

        tileGrid.display(fig, ax, show=False)
        return ax.get_children()

    ani = animation.FuncAnimation(fig, update, interval=300)
    plt.show()
    return ani

def HalfPlateExample():
    n = 30
    grid = None

    dE_func = lambda i, j: 2 * grid.grid[i][j] if i < n // 2 else -2 * grid.grid[i][j]

    grid = ExternalGrid(n, t_func=2.0, dE_func=dE_func)

    ims = []
    fig = plt.figure()

    for i in range(100):
        for j in range(1000):
            grid.metro()

        ims.append([plt.imshow(grid.grid, clim=(0, 1)), plt.text(0.9, 1.2, i)])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=0)

    ani.save("series.gif", writer=PillowWriter(fps=10))

def FindTPerc(tg):
    siz = len(tg.lis)
    
    for T in np.linspace(10.0, 0.1, num=100):
        n = 0
        for attempt in range(10):
            for t in tg.lis:
                t.spin = -1

            tg.redT = T
            
            while tg.wolff() == 0:
                ...

            cnt = sum(1 for t in tg.lis if t.spin > 0)

            if cnt / siz > 0.95:
                n += 1

        if n >= 8:
            return T

    return 0.1

def AutoSearch(new_grid, std_ratio=5, irreg_low=0.0016, irreg_high=0.005, T_low=1.5, T_high=15.0, avg_n=1):
    T_res = 1.0
    trial_length = 20
    E_samples = 100
    sample_step_factor = 2.0

    T_perc = FindTPerc(new_grid())
    print(T_perc)
    T_low = max(T_low, T_perc)

    size = len(new_grid().lis)
    
    def Irregularity(samples): # if irregular and not simple within submaxrange then increase sample_step
        tval = np.sum(np.diff(samples, n=2) ** 2)
        return tval

    def Simplicity(samples): # if simple then not irregular
        d = np.diff(samples)

        return np.sum(d[1:] * d[:-1] < 0)

    def SubMaxRange(samples, factor=0.7, scale=1.1): # if not narrowing down using sigma, narrow down using submaxrange
        m = samples.max()
        argmax = samples.argmax()
        w = np.argwhere(samples > m * factor)

        low = w.min()
        high = w.max()
        width = high - low

        if width < 3:
            width = 3

        low = argmax - scale * width
        high = argmax + scale * width

        low = max(0, int(low))
        high = min(len(samples) - 1, int(high))

        return low, high

    def Resolution(res, std, ratio=std_ratio): # if std < res decrease step, otherwise aim for std / res ~ ratio
        if std < res:
            return res / 5

        return std / 5

    run = True

    prev_std = [100, 100, 100, 100, 100]

    plt.ion()

    while run:
        correcting = True
        
        while correcting:
            print("correcting")

            T = np.arange(T_low, T_high + 1, T_res)            
            correcting = False

            C = []

            for i in range(avg_n):
                _, C_i = FindCriticalTemp(new_grid(), T, 15, E_samples, sample_step_factor, write_to_file=False)

                C += [np.array(C_i)]

            C = np.mean(C, axis=0)

            low, high = SubMaxRange(C)

            simple = Simplicity(C[low:high])

            print(simple)
            print(simple / (high - low))

            if simple / (high - low) > 0.4:
                correcting = True
                sample_step_factor *= 1.5
                E_samples *= 1.5
                E_samples = int(E_samples)
                avg_n += 1
                print("step up", sample_step_factor, E_samples, avg_n)

            if simple / (high - low) < 0.3 and sample_step_factor * size > 10:
                correcting = True
                sample_step_factor /= 2

                if E_samples > 100:
                    E_samples /= 2
                    E_samples = int(E_samples)

                if avg_n > 1:
                    avg_n -= 1

                print(low)
                print(high)
                print(T[low])
                print(T[high])

                T_low = T[low]
                T_low = max(T_perc, T_low)
                T_high = T[high]
                
                print("step down", sample_step_factor, E_samples, avg_n)

            if not correcting:
                T_low = T[low]
                T_low = max(T_perc, T_low)
                T_high = T[high]

            plt.figure()
            plt.plot(T, C)
            plt.show(block=False)
            plt.pause(0.001)

        print("testing")
        
        T = np.arange(T_low, T_high + 1, T_res)  
        T_crit = TrialCriticalTemp(new_grid, trial_length, T, 15, E_samples, sample_step_factor)

        if T_crit.std_dev > T_res * 5:
            if T_crit.std_dev * 1.1 > np.mean(prev_std):
                print("done")
                return T_crit

            prev_std += [T_crit.std_dev]

            del prev_std[0]

            T_low = T_crit.nominal_value - 2 * T_crit.std_dev
            T_low = max(T_perc, T_low)
            T_high = T_crit.nominal_value + 2 * T_crit.std_dev

        T_res = Resolution(T_res, T_crit.std_dev)

        print(T_crit)

from functools import partial
new = partial(Create666, depth=4)
AutoSearch(new)
