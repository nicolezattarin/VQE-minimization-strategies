#!/home/nicolezattarin/miniconda3/bin/python
import argparse
import time
import numpy as np
import qibo
import os
from aavqe import*
from ansatz import*
from traininglayer import *
from qibo import gates, models, hamiltonians
import tensorflow as tf
import myvqe
qibo.set_threads(1)


def VQEsimulation (nQubits,
                     nLayers,
                     VarLayer=False,
                     Method="Powell",
                     seed=0, ntrials=1000):
    """
        Performs a VQE circuit minimization test
    Args:
        nQubits (int): number of qubits to use in the simulation.
        nLayers (int): number of layers.
        VarLayer (bool): if True variational ansantz
                          is created using VariationalLayer.
        seed (int): seed for random parameters.
        ntrials (int): number of trials for hyperopt optimizer.

        Returns:
                Time
                Accuracy: log10 scaled
    """
    if Method == "sgd":
        qibo.set_backend("matmuleinsum")

    StartTime=time.time()
    
    #Create the variaional ansatz
    if VarLayer:
        circuit = VarLayerCircuit(nQubits, nLayers)
    else:
        circuit = StandardCircuit(nQubits, nLayers)
    
    #Hamiltonian: Heisenberg XXZ model
    Hamiltonian = hamiltonians.XXZ(nqubits=nQubits)
    VQE = models.VQE(circuit, Hamiltonian)
    _VQE = myvqe.myVQE(circuit, Hamiltonian)

    #Expected result
    Expected = np.real(np.min(Hamiltonian.eigenvalues().numpy()))

    #Set parameters
    np.random.seed(seed)

    #Optimization
    nParams = 2*nQubits*nLayers+nQubits
    InParameters = np.random.uniform(0, 2*np.pi, nParams)

        
    if Method=="migrad":
        Best, Params = _VQE.minimize(InParameters, method=Method)
        
    elif Method=="genetic":
        CXPB=0.5
        MUTPB=0.5
        MUTINDPB=0.5
        PopSize=100
        MaxGen=300
        options={'PopSize':PopSize, 'MaxGen':MaxGen,
                 'cxpb':CXPB ,'mutpb':MUTPB,'mutindpb':MUTINDPB}
        Best, Params = _VQE.minimize(InParameters, method=Method, options=options)
    
    elif Method == "bipop":
        Best, Params = _VQE.minimize(InParameters, method=Method, options=None)

    elif Method=="hyperopt":
        nTrials=ntrials
        studyname = "hopt_"+str(nQubits)+"qub_"+str(nLayers)+"lay"

        options={'nTrials':nTrials, 'studyname': studyname, 'verbose' : True}
        Best, Params = _VQE.minimize(InParameters, method=Method, options=options)
        
    elif Method=="spsa":
        options={'eta':0.4,
                 'eps':1e-4,
                 'maxiter':10000,
                 'etacorrection':None,
                 'alpha':0.101,
                 'gamma': 0.602 ,
                 'precision': 1e-10}
        Best, Params = _VQE.minimize(InParameters, method=Method, options=options)
        
    elif Method == "sgd":
        options = {'optimizer': "Adagrad"}
        Best, Params = VQE.minimize(InParameters, method=Method,
                                        options=options, compile=False)
    else:
        if (Method=="cma"):
            options=None #{'tolfun': 10**(-4)} #v termination criterion
            Best, Params = VQE.minimize(InParameters, method=Method,
                                        options=options, compile=False)
        else:
            options = {'disp': True}
            Best, Params = VQE.minimize(InParameters, method=Method,
                                        options=options, compile=False)
    #Accuracy-time
    Time = time.time()-StartTime
    Accuracy = np.log10(1/np.abs(Best-Expected))
    return Time, Accuracy


def AAVQESimulation(nqubits, nlayers, niter, T, seed, method):
    """
        Performs AAVQE circuit minimization test,
        see https://arxiv.org/abs/1806.02287v1 for details
    Args:
        nqubits (int): number of qubits to use in the simulation.
        nlayers (int): number of layers.
        niter (int): number of iterations.
        T (float): total rime.
        seed (int):seed for random number generation.
        method (string): optimization method for each step.

        Returns:
                Time
                Accuracy: log10 scaled
    """
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

def singlelayer_simulation (nqubits, nlayers, seed, method):
    """
        Performs single layer training optimization.
    Args:
        nqubits (int): number of qubits to use in the simulation.
        nlayers (int): number of layers.
        seed (int):seed for random number generation.
        method (string): optimization method for each step.

        Returns:
                Time
                Accuracy: log10 scaled
    """
    
    StartTime = time.time()
    # set the problem
    h=hamiltonians.XXZ(nqubits)
    circuit = StandardCircuit(nqubits, nlayers)
    
    
    np.random.seed(seed)
    nparams = 2*nqubits*nlayers+nqubits
    params_per_layer = 2*nqubits
    params = np.random.uniform(0, 2*np.pi, nparams)
    
    mytrain=SingleLayerOptimize(h, circuit,
                                params_per_layer, nlayers)
                                
    best, params = mytrain.minimize(params, method=method)

    expected=np.real(np.min(h.eigenvalues().numpy()))

    Time = time.time()-StartTime
    Accuracy = np.log10(1/np.abs(best-expected))
    
    return Time, Accuracy
