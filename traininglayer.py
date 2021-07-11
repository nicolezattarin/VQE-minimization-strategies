"""
Nicole Zattarin
layer by layer optimizing approach
"""

#!/home/nicolezattarin/miniconda3/bin/python

import numpy as np
import qibo
from qibo import gates, models, hamiltonians, optimizers
qibo.set_threads(1)
import logging
logging.basicConfig(level=logging.NOTSET)

import os, sys
sys.path.insert(0, os.path.abspath("optimizers"))
from ansatz import*
import optimizer



                
class SingleLayerOptimize():
    """
    Find the ground state of an hamiltonian using a variational circuits.
    In this approach we optimize single layers, fixing the rest of the trainable
    elements of the ansatz. We repeated these single-parameter and single-layer
    optimization cycles until we reached convergence.
    
    args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): hamiltonian
        circuit (:class:`qibo.abstractions.circuit.AbstractCircuit): variational ansatz.
        params_per_layer (int): number of params per layer.
        nlayers (int): number of layers of variational circuit provided.
        
    """
    
    def __init__ (self, hamiltonian, circuit, params_per_layer, nlayers):
    
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.params_per_layer = params_per_layer
        self.nlayers = nlayers

    def loss(self, varparams, fixed_right, fixed_left, hamiltonian, circuit):
            params = np.concatenate((fixed_left, varparams, fixed_right))

            circuit.set_parameters(params)
            FinalState = circuit()
            return hamiltonian.expectation(FinalState).numpy()
        
    def  minimize (self, params, method = "Trust-constr", options = None, precision = 1e-8):
        """
        args:
            params (np.array): : initial guess for the parameters.
            method (string): scipy method of optimization, or sgd
            options (dictionary): a dictionary with options for the different optimizers.
            precision: precision required for parameters.
            
        returns: best, params
        """
        from qibo import optimizers
        sys.path.insert(0, os.path.abspath("optimizers"))
        import optimizer

        oldparams = params*100

        while np.linalg.norm(oldparams-params)/np.linalg.norm(oldparams)>precision:
            oldparams = params
            # layers

            for i in range (1, self.nlayers+1):
                fixed_left = params[ : ( self.params_per_layer * (i-1) ) ]
                fixed_right = params[ (self.params_per_layer * (i) ) :]
                varparams = params[ ( self.params_per_layer * (i-1) ):\
                                    ( self.params_per_layer * (i) ) ]
                
                if  method == "migrad" or method == "genetic" or\
                    method == "spsa" or  method =="bipop" or \
                    method =="hyperopt" or method =="isres" or\
                    method == "ags" or method =="pso" or method =="shgo":
                    best, parameters = optimizer.optimize(self.loss, varparams,
                                                            args =(self.hamiltonian, self.circuit),
                                                            method=method)

                else
                    best, bestparams = optimizers.optimize(self.loss, varparams,
                                                                    args=(fixed_right, fixed_left,
                                                                    self.hamiltonian, self.circuit),
                                                                    method=method)
                                                                 
                params = np.concatenate((fixed_left, bestparams, fixed_right))
                logging.info("\nTraining layer " + str(i) + " terminated with vlue: " + str(best) +\
                             "\nparams:\n" + str(params))
                             
            #core circuit
            fixed_left = params[: self.nlayers*self.params_per_layer]
            fixed_right = np.array([])
            varparams = params[self.nlayers*self.params_per_layer:]


            if  method == "migrad" or method == "genetic" or\
                method == "spsa" or  method =="bipop" or \
                method =="hyperopt" or method =="isres" or\
                method == "ags" or method =="pso" or method =="shgo":
                best, parameters = optimizer.optimize((self.loss, varparams,
                                                        args=(self.hamiltonian, self.circuit),
                                                        method=method)

            else
                best, bestparams = optimizers.optimize(self.loss, varparams,
                                                                args=(fixed_right, fixed_left,
                                                                self.hamiltonian, self.circuit),
                                                                method=method)
            params = np.concatenate((fixed_left, bestparams, fixed_right))
            logging.info("\nTraining final parameters terminated with vlue: " + str(best) +\
                             "\nparams:\n" + str(params))

            self.circuit.set_parameters(params)
        
        return best, params
