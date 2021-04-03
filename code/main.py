#!/home/nicolezattarin/miniconda3/bin/python
import argparse
import time
import numpy as np
import qibo
import os
import matplotlib.pylab as plt
from vqe_tester import *
from aavqe_tester import *
from qibo import gates, models, hamiltonians
import tensorflow as tf
qibo.set_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=2, help="Number of qubits.", type=int)
parser.add_argument("--startnlayers", default=1, help="Starting nlayers.", type=int)
parser.add_argument("--nstep", default=10, help="Number of steps.", type=int)
parser.add_argument("--dir", default="results", help="Directory to save data.", type=str)
parser.add_argument("--varlayer", default=False, help="Use VariationalLayer gate.")
parser.add_argument("--method", default="Powell", help="Method of minimization.", type=str)
parser.add_argument("--seed", default=0, help="seed for initial parameters.", type=int)
parser.add_argument("--ntrials", default=10000, help="trials for hyperopt minimization.", type=int)



def main  (nqubits,
           startnlayers,
           nstep,
           dir,
           varlayer,
           method,
           seed,
           ntrials):
    """
    Creates a file: "dir/method-nqubits-qubits.txt"
       nsteps number of points (time, accuracy) starting from startnlayers, step=1
       Accuracy measured as log10(1/err)

    Args:
       - nqubits (int): number of qubits to use in the simulation.
       - startnlayers (int): number of starting layers.
       - nstep (int): number of increases of startnlayers.
       - dir (str): directory to save files.
       - varlayer (bool): if True variational ansants
                          is created using VariationalLayer.
       - method (str): methods of minimization.
    """
    
    Depth=np.arange(startnlayers,startnlayers+nstep)
    time = np.zeros(nstep)
    accuracy = np.zeros(nstep)
        
    if method == "aavqe":
        for i in range(nstep):
            # here i set the best iter/T as results from testing accuracy-time vs iterations
            iter=10
            T=1
            aavqe_method="COBYLA"
            time[i], accuracy[i], = AAVQESimulation(nqubits, Depth[i], iter, T, seed, aavqe_method)

    else:
        for i in range(nstep):
            accuracy[i], time [i] = MinimizationTest(nqubits, Depth[i],
                                                     varlayer, method, seed, ntrials)

        if (os.path.isdir(dir)==False):
            os.mkdir(dir)
        
    nQubitStr=str(nqubits)
    MethodStr=str(method)
    file=open(dir+"/"+MethodStr+nQubitStr+"qubits.txt", "w")
    file.write("nLayers\ttime\taccuracy\n")
    for i in range(len(time)):
            file.write(str(Depth[i]))
            file.write("\t")
            file.write(str(time[i]))
            file.write("\t")
            file.write(str(accuracy[i]))
            file.write("\n")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
