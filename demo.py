from tests import Create4444, Create666, AnimateTile, ShowAnimate, FindCriticalTemp,\
Create333333, Create3636, Create46_12, TrialCriticalTemp, ShowPhaseTransitionNewerThan
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import datetime
#Exp2_6_1, Exp2_6_2, Exp2_6_3, Exp2_6_4, Exp2_6_5

if True:
    #print("Exploratory questions")
    #print("2.6.1")
    #Exp2_6_1()
    ##Exp2_6_2()
    #Exp2_6_3()
    #Exp2_6_4()
    #print("2.6.5")
    #Exp2_6_5()
    
    ani1 = ShowAnimate(40, 2.3, 100, 10)
    plt.pause(5)
    ani2 = AnimateTile(Create666(4))
    plt.pause(5)
    ani3 = AnimateTile(Create4444(4))
    plt.pause(5)
    T_crit = FindCriticalTemp(Create4444(10), show=True, write_to_file=False)
    print("Critical temperature of diamond (skewed square) grid of radius 10:", T_crit)
    plt.pause(5)


generators = [Create333333, Create3636, Create4444, Create666, Create46_12]
generators = [partial(g, depth=4) for g in generators]

if True:
    print("Fast tests")
    print("For these T_crits, the resolution of T dominates the external error")
    for g in generators:
        print("T_crit of ", g().createID, ": ", end='')
        T = np.linspace(1.2, 4.0, num=20)
        T_crit = TrialCriticalTemp(g, T=T, trial_length=5, settle_factor=2.0, E_samples=20, sample_step_factor=1.0, write_to_file=False)
        
        T_res = (4.0 - 1.2) / 20
        T_crit.std_dev = T_res
        print(T_crit)

if False:
    for g in generators:
        tg = g()
        
        for t in tg.lis:
            t.spin = -1
        
        fig = plt.figure(figsize=(12, 12), dpi=80)
        
        fig, ax = tg.display(fig=fig)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        fig.tight_layout()
        
        fig.savefig(str(tg.createID) + ".pdf")
        
if True:
    data = np.genfromtxt(r"PhaseTransitionData\Create333333_10_12110501_0.txt", delimiter=",\t")
    
    T = data.T[0]
    E = data.T[1]
    C = data.T[2]
    m = data.T[3]
    
    E_err = T * C ** 0.5
    m_abs = abs(m)
    
    fig = plt.figure(figsize=(12, 12), dpi=100)
    axE, axC, axm = fig.subplots(nrows=3)
    
    axE.plot(T, E, 'ro')
    axE.errorbar(T, E, yerr=E_err, fmt='none')
    axC.plot(T, C, 'ro')
    axm.plot(T, m_abs, 'ro')
    
    axE.set_ylabel("$E$")
    axC.set_ylabel("$C$")
    axm.set_ylabel("$|m|$")
    axm.set_xlabel(r"$T_{red}$")
    
    fig.suptitle("A single trial of 3.3.3.3.3.3")
    
    fig.savefig("single trial")
    fig.show()
    
if True:
    ShowPhaseTransitionNewerThan(datetime.datetime(2019,9,25), yAx=["C", "E", "CDens"])