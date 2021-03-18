#!/home/nicolezattarin/miniconda3/bin/python

import numpy as np
import qibo
from qibo import gates, models, hamiltonians
import myMinimizers

class myVQE(object):

    def __init__(self, circuit, hamiltonian):
#       Initialize circuit ansatz and hamiltonian.
        self.circuit = circuit
        self.hamiltonian = hamiltonian

    def minimize(self, InParams, method="migrad", options=None):
    
#    Performs minimization to find the ground state
#    Args:
#        - InParams (np.ndarray): Initial guess for the variational parameters.
#        - method (str): Name of optimizer to use. migrad for iMinuit.
#        - options (dictionary): additional settings.

        def loss(params, hamiltonian, circuit):
        # returns the expectation value of hamiltonian in the final state.
            circuit.set_parameters(params)
            FinalState = circuit()
            return hamiltonian.expectation(FinalState).numpy()
                                                               

        best, parameters = myMinimizers.optimize(loss, InParams,
                                                 args=(self.hamiltonian, self.circuit),
                                                 method=method, options=options)
        
        
        self.circuit.set_parameters(parameters)
        return best, parameters

