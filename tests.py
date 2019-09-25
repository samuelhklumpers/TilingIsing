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

    tile = TileGrid(hex_constr, depth=depth, seed=seed, createdID="Create3636")

    return tile

def Create333333(depth, seed=None):
    tri_constr = TilingConstraint(3)

    tri_constr.set_neighbours([tri_constr], 3)

    tri_constr.set_constraint(tri_constr, 1, -1, -1, -1, -1, -1)
    tri_constr.set_constraint(tri_constr, -1, 1, 1, 1, 1, 1)

    tile = TileGrid(tri_constr, depth=depth, seed=seed, createID="Create333333")

    return tile

def Create555(depth, seed=None):
    tri_constr = TilingConstraint(5)

    tri_constr.set_neighbours([tri_constr], 5)

    tri_constr.set_constraint(tri_constr, 1, -1, -1)
    tri_constr.set_constraint(tri_constr, -1, 1, 1)

    tile = TileGrid(tri_constr, depth=depth, seed=seed, createID="Create555")

    return tile

def Create33333(depth, seed=None):
    tri_constr = TilingConstraint(3)

    tri_constr.set_neighbours([tri_constr], 3)

    tri_constr.set_constraint(tri_constr, 1, -1, -1, -1, -1)
    tri_constr.set_constraint(tri_constr, -1, 1, 1, 1, 1)

    tile = TileGrid(tri_constr, depth=depth, seed=seed, createID="Create33333")

    return tile

def Create4444(depth, seed=None):
    sq_constr = TilingConstraint(4)

    sq_constr.set_neighbours([sq_constr], 4)

    sq_constr.set_constraint(sq_constr, 1, -1, -1, -1)
    sq_constr.set_constraint(sq_constr, -1, 1, 1, 1)

    tile = TileGrid(sq_constr, depth=depth, seed=seed, createID="Create4444")

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

def FindCriticalTemp(grid, T=np.linspace(1, 10), settle_t=100,
                    E_samples=100, sample_step=100, show=False,
                    write_to_file=True, trial_seed=None, trial_index=None,
                    trial_length=None):
    E = []
    m = []
    C = []

    for temp in T:
        grid.redT = temp
        
        for _ in range(settle_t):
            grid.wolff()

        E2 = []

        for _ in range(E_samples):
            for _ in range(sample_step):
                grid.wolff()
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
        trial_seed_val = trial_seed if trial_seed is not None else "null"
        trial_index_val = trial_index if trial_index is not None else "null"
        trial_length_val = trial_length if trial_length is not None else "null"
        
        file.write("# PARAM_DICT {}\n".format(json.dumps({"genRepr": genRepr, "settle_t": settle_t,
                   "E_samples": E_samples, "sample_step": sample_step,
                   "trial_seed": trial_seed_val,
                   "trial_index": trial_index_val,
                   "trial_length": trial_length_val})))
        file.write(f"# CRIT_TEMP_NAIVE {critTempNaive:.4e}\n")
        file.write(f"# PARAM GENREPR {genRepr}\n")
        file.write(f"# PARAM SETTLE_T {settle_t}\n")
        file.write(f"# PARAM E_SAMPLES {E_samples}\n")
        file.write(f"# PARAM SAMPLE_STEP {sample_step}\n")
        file.write(f"# PARAM TRIAL_SEED {trial_seed_val}\n")
        file.write(f"# PARAM TRIAL_INDEX {trial_index_val}\n")
        file.write(f"# PARAM TRIAL_LENGTH {trial_length_val}\n")
                   
        file.write(f"# FORMAT T, E, C, m\n")
        dat = np.transpose(np.array([T, E, C, m]))
        for row in dat:
            file.write(",\t".join(["{:.4f}".format(a) if 0.01 <= abs(a) <= 1000 
                                   else "{:.4e}".format(a) for a in row]) + "\n")
        file.close()

    return critTempNaive

def TrialCriticalTemp(new_grid, trial_length=20, T=np.linspace(1, 10), settle_t=100,
                        E_samples=100, sample_step=100, trial_seed=None):
    if trial_seed is None:
        gen = random.RandomState()
    else:
        gen = random.RandomState(trial_seed)
        
    seeds = gen.randint(0, 1e8, trial_length)

    T_crits = [FindCriticalTemp(new_grid(seed=seeds[i]), T=T, settle_t=settle_t, 
                                E_samples=E_samples, sample_step=sample_step, 
                                show=False, trial_seed=trial_seed, trial_index=i,
                                trial_length=trial_length)
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


