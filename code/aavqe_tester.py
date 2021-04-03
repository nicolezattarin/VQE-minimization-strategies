#!/home/nicolezattarin/miniconda3/bin/python
import argparse
import time
import numpy as np
import qibo
import os
from aavqe import*
from ansatz import*
from qibo import gates, models, hamiltonians
import tensorflow as tf
qibo.set_threads(1)


def AAVQESimulation(nqubits, nlayers, niter, T, seed, method):

    StartTime = time.time()
    
    # set the problem
    h0=hamiltonians.X(nqubits)
    h1=hamiltonians.XXZ(nqubits)
    circuit = StandardCircuit(nqubits, nlayers)
    
    #set scheduling function
    s = lambda t: t
    
    dt = T/niter
    np.random.seed(seed)
    nparams = 2*nqubits*nlayers+nqubits
    params = np.random.uniform(0, 2*np.pi, nparams)
    
    myaavqe=AAVQE(circuit, h0, h1, s, dt, T)
    best, params = myaavqe.minimize(params, method)
    
    expected=np.real(np.min(h1.eigenvalues().numpy()))

    Time = time.time()-StartTime
    Accuracy = np.log10(1/np.abs(best-expected))

    return Time, Accuracy
    
