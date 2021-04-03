import argparse
import time
import numpy as np
import qibo
from qibo import gates, models, hamiltonians
from ansatz import*
import myvqe


def MinimizationTest(nQubits,
                     nLayers,
                     VarLayer=False,
                     Method="Powell",
                     seed=0, ntrials=10000):
    """
        Performs a VQE circuit minimization test
    Args:
       - nQubits (int): number of qubits to use in the simulation.
       - nLayers (int): number of layers.
       - VarLayer (bool): if True variational ansantz
                          is created using VariationalLayer.
       - seed (int): seed for random parameters.
       - ntrials (int): number of trials for hyperopt optimizer.

        Returns:
                - Accuracy: log10 scaled
                - Time
    """

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
        
    elif Method=="hyperopt":
        nTrials=ntrials
        options={'nTrials':nTrials}
        Best, Params = _VQE.minimize(method=Method,nqubits=nQubits,
                                     nlayers=nLayers, options=options)
    else:
        if (Method=="cma"):
            options={'tolfun': 10**(-4)} #v termination criterion
            Best, Params = VQE.minimize(InParameters, method=Method,
                                        options=options, compile=False)
        else:
            options = {'disp': True}
            Best, Params = VQE.minimize(InParameters, method=Method,
                                        options=options, compile=False)
    #Accuracy-time
    Time = time.time()-StartTime
    Accuracy = np.log10(1/np.abs(Best-Expected))
    
    return Accuracy,Time


