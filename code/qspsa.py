import numpy as np
from qibo import models, callbacks, gates, hamiltonians
from qibo.abstractions import hamiltonians
import logging
from scipy import linalg
from 

class qSPSA (object):
    """
    Quantum Simultaneous Perturbation Stochastic Approximation.
    See https://arxiv.org/abs/2103.09232
    
    Args:
        - circuit (:class:`qibo.abstractions.circuit.AbstractCircuit`): Circuit that
            implements the variaional ansatz.
        - hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object.
    """

    def __init__ (self, circuit, hamiltonian):
        if not issubclass(type(hamiltonian), hamiltonians.HAMILTONIAN_TYPES):
            raise_error(TypeError, "h0 should be a hamiltonians.Hamiltonian "
                                  "object but is {}.".format(type(hamiltonian)))
        
        self.circuit = circuit
        self.hamiltonian = hamiltonian

    def minimize (self, params0, nqubits,
                        seed=0, eps=1e-5,
                        maxiter=1e4, eta=1e-1,
                        beta=1e-3):
        """
        Mimimizer to find the ground state of hamiltonian provided
        Args:
            - params (np.array or list): initial guess for circuit parameters,
                                          may be chosen randomly
            - seed (int): seed for random samplings
            - eps (float): step size of expansion
            - maxiter (int): maximum number of iterations.
            - eta (float): learning rate.
            - beta (float): small positive constant to obtain the regularization of Hessian.
        """

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
        oldbest = lossf(params0, self.hamiltonian, self.circuit)
        
        while np.linalg.norm(oldparams-params)/np.linalg.norm(oldparams)>1e-8\
              and iter < maxiter:

            # The optimisation carried out until the solution has converged, or
            # the maximum number of itertions has been reached.
            
            oldparams = params # Store theta at the start of the iteration
            
            # define perturbation
            np.random.seed(seed)
            delta1 = np.random.uniform(-1, 1, len(params))
            delta2 = np.random.uniform(-1, 1, len(params))

            # Define the state
            # Calculation of the absolute value of the overlap of |psi_t>
            # with parameter values t and slightly shifted parameters t+eps*delta
            # |<psi_{t+eps*delta}|psi_t>|^2
            
            U.set_parameters(params)
            # define state |0> that is the bra for overlap
            state0=np.zeros(2**nqubits)
            state0[0]=1
            overlap = callbacks.Overlap(state0)

            Udag.set_parameters(params+eps*delta1+eps*delta2)
            c1 = Udag + U
            c1.add(gates.CallbackGate(overlap))
            c1()
            Udag.set_parameters(params+eps*delta1)
            c2 = Udag + U
            c2.add(gates.CallbackGate(overlap))
            c2()
            Udag.set_parameters(params-eps*delta1+eps*delta2)
            c3 = Udag + U
            c3.add(gates.CallbackGate(overlap))
            c3()
            Udag.set_parameters(params-eps*delta1)
            c4 = Udag + U
            c4.add(gates.CallbackGate(overlap))
            c4()

            dF = overlap[:].numpy()
            deltaF = dF[0]**2-dF[1]**2-dF[2]**2+dF[3]**2
            
#            psi = params
#            phi1 = params+eps*delta1+eps*delta2
#            phi2 = params+eps*delta1
#            phi3 = params-eps*delta1+eps*delta2
#            phi4 = params-eps*delta1
#            deltaF = np.dot(psi,phi1)**2 + np.dot(psi,phi2)**2\
#                   + np.dot(psi,phi3)**2 + np.dot(psi,phi4)**2
            

#
#            deltaF = + lossf (params + eps*delta1 + eps*delta2, self.hamiltonian, self.circuit)\
#                     - lossf(params + eps*delta1, self.hamiltonian, self.circuit)\
#                     - lossf(params - eps*delta1 + eps*delta2, self.hamiltonian, self.circuit)\
#                     + lossf(params - eps*delta1, self.hamiltonian, self.circuit)
            
            # calculate g metric:
            # G_k = k/(k+1) G_(k-1) + 1/(k+1) g_k
            # note that we use matrix-square-root ( Gbar * Gbar )
            # so that G is semi-def>0
            # to ensure invertibility of the Hessian estimate we further add a small
            # positive constant to the diagonal and obtain the regularization
            # matrix-square-root ( Gbar * Gbar ) + id * beta

            if iter == 0:
                gbar = -(1./8.) * (deltaF/eps**2)\
                       * (np.dot(delta1,delta2) + np.dot(delta2,delta1))
                       
                gbarmat = gbar * np.identity(len(params))#,len(params)))
                G = linalg.sqrtm (gbarmat*gbarmat) + beta*np.identity(len(params))

            else:
                ghat = -(1./8.) * (deltaF/eps**2)\
                       * (np.dot(delta1,delta2) + np.dot(delta2,delta1))
                       
                gbar = 1/(iter+1) * (iter * oldgbar + ghat)
                gbarmat = gbar * np.identity(len(params))#,len(params)))
                G = linalg.sqrtm (gbarmat*gbarmat) + beta*np.identity(len(params))
                #print (G[0,0], G[1,0])
                
            # calculate gradient with first-order SPSA
            # gradf = ( f(r + eps * delta) - f(r - eps * delta) ) / (2 * eps)
            
            delta = np.random.uniform(-1, 1, len(params))
            gradf = (0.5 /eps) * \
                    (lossf(params+eps*delta, self.hamiltonian, self.circuit)\
                    -lossf(params-eps*delta, self.hamiltonian, self.circuit))*delta
            
 
            # update
            tempparams = oldparams - eta * np.dot(np.linalg.inv(G),gradf)
            best = lossf(tempparams, self.hamiltonian, self.circuit)
            
            # blocking condition that only accepts update steps if the loss
            # at the candidate parameters is smaller than the current loss, plus a tolerance
            ATOL = 1e-3
            if best>oldbest:
                oldparams=100*params
                print("salta iterazione")
                continue
            else:
                params = tempparams
                oldgbar = gbar
                iter += 1
                oldbest=best
            
                logging.basicConfig(level=logging.NOTSET)
                logging.info("Iteration "+ str(iter-1)+ " terminated with value "+ str(best)+
                             "\nparams \n"+ str(params))
            
        return best, params, iter

