import numpy as np
from qibo import models, callbacks, gates
import logging

class qSPSA (object):
    """
    Quantum Simultaneous Perturbation Stochastic Approximation.
    
    """

    def __init__ (self, circuit, hamiltonian,
                  epsilon=1e-2, maxiter=1e4, eta=1e-3):
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.eps = epsilon
        self.maxiter = maxiter
        self.eta = eta
        

    def minimize (self, params0, nqubits, seed=0, args=()):

        def lossf(params, hamiltonian, circuit):
        # returns the expectation value of hamiltonian in the final state.
            circuit.set_parameters(params)
            FinalState = circuit()
            return hamiltonian.expectation(FinalState).numpy()

        iter = 0
        params = params0
        oldparams = params0*100
        
        U = self.circuit
        Udag = U.invert()
        
        while (iter < self.maxiter):
            # The optimisation carried out until the solution has converged, or
            # the maximum number of itertions has been reached.
            
            oldparams = params # Store theta at the start of the iteration
            
            # define perturbation
            np.random.seed(seed)
            delta1 = np.random.uniform(-1, 1, len(params))
            delta2 = np.random.uniform(-1, 1, len(params))
            delta = np.random.uniform(-1, 1, len(params))

            # define the state
            U.set_parameters(params)
            overlap = callbacks.Overlap(np.zeros(2**nqubits))

            Udag.set_parameters(params+self.eps*delta1+self.eps*delta2)
            c1 = Udag + U
            print(c1.summary())
            c1.add(gates.CallbackGate(overlap))
            c1()
            Udag.set_parameters(params+self.eps*delta1)
            c2 = Udag + U
            c2.add(gates.CallbackGate(overlap))
            c2()
            Udag.set_parameters(params-self.eps*delta1+self.eps*delta2)
            c3 = Udag + U
            c3.add(gates.CallbackGate(overlap))
            c3()
            Udag.set_parameters(params-self.eps*delta1)
            c4 = Udag + U
            c4.add(gates.CallbackGate(overlap))
            c4()
            
            dF = overlap[:].numpy()
            deltaF = dF[0]-dF[1]-dF[2]+dF[3]
            print(dF)

            
            # calculate g metric:
            # G_k = k/(k+1) G_(k-1) + 1/(k+1) g_k
            if iter == 0:
                G = -(1./6.) * (deltaF/self.eps**2) * np.dot(delta1,delta2) * np.dot(delta2,delta1)
            else:
                g = -(1./6.) * (deltaF/self.eps**2) * np.dot(delta1,delta2) * np.dot(delta2,delta1)
                G = 1/(iter+1) * (iter * oldG + g)
            
            #AGGIUNGI AGGIONAMENTO DI G
            
            # calculate gradient with first-order SPSA
            
            gradf = 0.5 * (1/self.eps)*(lossf(oldparams+self.eps*delta, self.hamiltonian, self.circuit)+ lossf(oldparams-self.eps*delta, self.hamiltonian, self.circuit) )*delta
            
            # update
            params = oldparams - self.eta * G * gradf
            best = lossf(params, self.hamiltonian, self.circuit)
            oldG = G
            iter += 1
            
            logging.basicConfig(level=logging.NOTSET)
            logging.info("Iteration "+ str(iter-1)+ " terminated with value "+ str(best)+
                         "\nparams \n"+ str(params))
            
        return best, params, iter
