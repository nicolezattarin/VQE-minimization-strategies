import argparse
import time
import numpy as np
import qibo
import seaborn as sns
import matplotlib.pylab as plt
from MinimizationTest import *
from qibo import gates, models, hamiltonians
from threading import Thread


parser = argparse.ArgumentParser()
parser.add_argument("--nQubits", default=2, help="Number of qubits.", type=int)
parser.add_argument("--nLayers", default=1, help="Starting nlayers.", type=int)
parser.add_argument("--Method", default="Powell", help="Method of minimization.", type=str)
parser.add_argument("--MaxIter", default=None, help="Maximum optimization iterations.", type=int)
parser.add_argument("--VarLayer", action="store_true", help="Use VariationalLayer gate.")

def main (nQubits, nLayers, VarLayer=False, MaxIter=None, Method="Powell"):

#       Creates a file: "Method-nQubits-qubits.txt"
#       # stepLayers points (time, accuracy) starting from nLayers, step=1
#       Accuracy measured as log10(1/err)
    
    stepLayers=10
    Depth=np.arange(nLayers,nLayers+stepLayers)

    time = np.zeros(stepLayers)
    accuracy =  np.zeros(stepLayers)

    for i in range(stepLayers):
        accuracy[i], time [i]=MinimizationTest(nQubits, nLayers, VarLayer, MaxIter, Method)
        nLayers+=1
    
    nQubitStr=str(nQubits)
    MethodStr=str(Method)

    temp=[time,accuracy]
    file=open("results/"+MethodStr+nQubitStr+"qubits.txt", "w")
    file.write("time\taccuracy\n")
    for i in range(len(time)):
            file.write(str(time[i]))
            file.write("\t")
            file.write(str(accuracy[i]))
            file.write("\n")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
