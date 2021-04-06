#!/home/nicolezattarin/miniconda3/bin/python
import argparse
import time
import numpy as np
import qibo
import os
from qspsa import*
from ansatz import*
from qibo import gates, models, hamiltonians
import tensorflow as tf
qibo.set_threads(1)

nqubits=2
nlayers=2

StartTime = time.time()

# set the problem
h=hamiltonians.XXZ(nqubits)
circuit = StandardCircuit(nqubits, nlayers)

#set scheduling function
np.random.seed(0)
nparams = 2*nqubits*nlayers+nqubits
params = np.random.uniform(0, 2*np.pi, nparams)

myspsa=qSPSA(circuit, h)
best, params, iter = myspsa.minimize(params, nqubits)

expected=np.real(np.min(h.eigenvalues().numpy()))

Time = time.time()-StartTime
Accuracy = np.log10(1/np.abs(best-expected))

print(Accuracy)
print(expected)

print(best)
print(iter)

