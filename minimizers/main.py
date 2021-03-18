import argparse
import time
import numpy as np
import qibo
import os
import matplotlib.pylab as plt
from tester import *
from qibo import gates, models, hamiltonians
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--nQubits", default=2, help="Number of qubits.", type=int)
parser.add_argument("--StartnLayers", default=1, help="Starting nlayers.", type=int)
parser.add_argument("--NStep", default=10, help="Number of steps.", type=int)
parser.add_argument("--Directory", default="results", help="Directory to save data.", type=str)
parser.add_argument("--VarLayer", default=False, help="Use VariationalLayer gate.")
parser.add_argument("--MaxGen", default=300, help="Maximum number of generations for genetic algorithm.", type=int)

parser.add_argument("--PopSize", default=100, help="size of the population for genetic algorithm.", type=int)

parser.add_argument("--Method", default="Powell", help="Method of minimization.", type=str)



def main  (nQubits,
           StartnLayers,
           NStep,
           Directory,
           VarLayer,
           MaxGen,
           PopSize,
           Method):

#    Creates a file: "Directory/Method-nQubits-qubits.txt"
#       nSteps number of points (time, accuracy) starting from StartnLayers, step=1
#       Accuracy measured as log10(1/err)
#
#    Args:
#       - nQubits (int): number of qubits to use in the simulation.
#       - StartnLayers (int): number of starting layers.
#       - NStep (int): number of increases of StartnLayers.
#       - Directory (str): directory to save files.
#       - VarLayer (bool): if True variational ansants
#                          is created using VariationalLayer.
#       - MaxGen (int): maximum number of generations as
#                       option for genetic algorithm.
#       - PopSize (int): size of the population as
#                        option for genetic algorithm.
#       - Method (str): methods of minimization.

        
    Depth=np.arange(StartnLayers,StartnLayers+NStep)
    time = np.zeros(NStep)
    accuracy = np.zeros(NStep)
    CurrentnLayers = StartnLayers
    
    for i in range(NStep):
        accuracy[i], time [i] = MinimizationTest(nQubits, Depth[i],
                                                 VarLayer, MaxGen, PopSize, Method)

    if (os.path.isdir(Directory)==False):
        os.mkdir(Directory)
        
    nQubitStr=str(nQubits)
    MethodStr=str(Method)
    file=open(Directory+"/"+MethodStr+nQubitStr+"qubits.txt", "w")
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
