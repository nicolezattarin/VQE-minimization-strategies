import argparse
import time
import numpy as np
import qibo
from qibo import gates, models, hamiltonians

#Creates variational circuit using normal gates.
def StandardCircuit (nQubits, nLayers):
    circuit = models.Circuit(nQubits)
    for l in range(nLayers):
        circuit.add((gates.RY(q, theta=0) for q in range(nQubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(0, nQubits-1, 2)))
        circuit.add((gates.RY(q, theta=0) for q in range(nQubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(1, nQubits-2, 2)))
        circuit.add(gates.CZ(0, nQubits-1))
    circuit.add((gates.RY(q, theta=0) for q in range(nQubits)))
    return circuit

"""Creates variational circuit using ``VariationalLayer`` gate."""
def VarLayerCircuit(nQubits, nLayers):
    circuit = models.Circuit(nQubits)
    pairs = list((i, i + 1) for i in range(0, nQubits - 1, 2))
    theta = np.zeros(nQubits)#params
    
    for l in range(nLayers):
        circuit.add(gates.VariationalLayer(range(nQubits), pairs,
                                           gates.RY, gates.CZ,
                                           theta, theta))
        circuit.add((gates.CZ(i, i + 1) for i in range(1, nQubits - 2, 2)))
        circuit.add(gates.CZ(0, nQubits - 1))
    circuit.add((gates.RY(i, theta) for i in range(nQubits)))
    
    return circuit



def MinimizationTest(nQubits, nLayers, VarLayer=False, MaxIter=None, Method="Powell"):

#        Performs a VQE circuit minimization test
#        Returns:
#                - Error: log10 scaled
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
    
    #Expected result
    Expected = np.real(np.min(Hamiltonian.eigenvalues().numpy()))

    #Set parameters
    np.random.seed(0)
    nParams = 2*nQubits*nLayers+nQubits
    InParameters = np.random.uniform(0, 2*np.pi, nParams)

    #Optimization
    if (Method!="trust-constr"):
        options = {'disp': True, 'MaxIter': MaxIter} #options depend on the optimizer
    else:
        options = {'disp': True}
        
    Best, Params = VQE.minimize(InParameters, method=Method,
                                options=options, compile=False)
                                
    #Accuracy-time
    Time = time.time()-StartTime
    Error = np.log10(1/np.abs(Best-Expected))
    
    return Error,Time


