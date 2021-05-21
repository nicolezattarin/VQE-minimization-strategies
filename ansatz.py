"""
Nicole Zattarin
Variational ansatz for VQE testing
"""

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

#Creates variational circuit using ``VariationalLayer`` gate.
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
