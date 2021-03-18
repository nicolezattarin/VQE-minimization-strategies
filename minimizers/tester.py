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
                     MaxGen=300,
                     PopSize=100,
                     Method="Powell"):

#        Performs a VQE circuit minimization test
#    Args:
#       - nQubits (int): number of qubits to use in the simulation.
#       - nLayers (int): number of layers.
#       - VarLayer (bool): if True variational ansantz
#                          is created using VariationalLayer.
#       - MaxGen (int): maximum number of generations as
#                       option for genetic algorithm.
#       - PopSize (int): size of the population as
#                        option for genetic algorithm.
#       - Method (str): methods of minimization.
#
#        Returns:
#                - Accuracy: log10 scaled
#                - Time

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
    np.random.seed(0)


    #Optimization
    nParams = 2*nQubits*nLayers+nQubits
    InParameters = np.random.uniform(0, 2*np.pi, nParams)

    
    if (Method=="migrad"):
        Best, Params = _VQE.minimize(InParameters, method=Method)
    elif (Method=="genetic"):
        options={'PopSize':PopSize, 'MaxGen':MaxGen}
        Best, Params = _VQE.minimize(InParameters, method=Method, options=options)
    else:
        if (Method=="cma"):
            Best, Params = VQE.minimize(InParameters, method=Method,
                                        options={}, compile=False)
        else:
            options = {'disp': True}
            Best, Params = VQE.minimize(InParameters, method=Method,
                                        options=options, compile=False)
    #Accuracy-time
    Time = time.time()-StartTime
    Error = np.log10(1/np.abs(Best-Expected))
    
    return Error,Time


