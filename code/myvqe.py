#!/home/nicolezattarin/miniconda3/bin/python

import numpy as np
import qibo
from qibo import gates, models, hamiltonians
import myMinimizers
from optuna import samplers, pruners, visualization
qibo.set_threads(1)

class myVQE(object):

    def __init__(self, circuit, hamiltonian):
#       Initialize circuit ansatz and hamiltonian.
        self.circuit = circuit
        self.hamiltonian = hamiltonian

    def minimize(self, InParams=[], method="migrad", options=None, nqubits=4, nlayers=4):
        """
        Performs minimization to find the ground state
        Args:
            - InParams (np.ndarray): Initial guess for the variational parameters.
            - method (str): Name of optimizer to use. migrad for iMinuit.
            - options (dictionary): additional settings.
            - nqubits (int): number of qubits, necessary only if hyperopt optimizer is used
            - nlayers (int): number of layers, necessary only if hyperopt optimizer is used
        """
        
        def loss(params, hamiltonian, circuit):
        # returns the expectation value of hamiltonian in the final state.
            circuit.set_parameters(params)
            FinalState = circuit()
            return hamiltonian.expectation(FinalState).numpy()
                
        
        if (method=="hyperopt"):
            best, parameters = myMinimizers.optimize(loss, nqubits=nqubits, nlayers=nlayers,
                                                    args=(self.hamiltonian, self.circuit),
                                                    method=method, options=options)
            parameters=list(parameters.values())
        else:
            best, parameters = myMinimizers.optimize(loss, InParams,
                                                    args=(self.hamiltonian, self.circuit),
                                                    method=method, options=options)
        
        
        self.circuit.set_parameters(parameters)
        return best, parameters

