import numpy as np
import qibo
from qibo import gates, models, hamiltonians
qibo.set_threads(1)
from ansatz import *
import logging
import logging.handlers
from qibo.abstractions import hamiltonians
from qibo.evolution import StateEvolution
logging.basicConfig(level=logging.NOTSET)


class AAVQE(StateEvolution):

    """This class implements the Adiabatically Assisted Variational Quantum Eigensolvers
 algorithm. See https://arxiv.org/abs/1806.02287
    Args:
        circuit (:class:`qibo.abstractions.circuit.AbstractCircuit`): The variaional ansatz.
        in_hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): easy hamiltonian.
        fin_hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): problem hamiltonian.
        scheduling_func (callable): Function of time that defines the scheduling of the
            adiabatic evolution. Note that in this implmentation parameterized scheeduling
            fucntions are not allowd.
        dt (float): Time step to use for the numerical integration of
            Schrondiger's equation.
        total time (float): total time of time evolution.

    """
    def __init__(self, circuit, in_hamiltonian,
                       fin_hamiltonian, scheduling_func,
                       dt, total_time):
        
        if not issubclass(type(in_hamiltonian), hamiltonians.HAMILTONIAN_TYPES):
            raise_error(TypeError, "h0 should be a hamiltonians.Hamiltonian "
                                  "object but is {}.".format(type(in_hamiltonian)))
        if type(fin_hamiltonian) != type(in_hamiltonian):
            raise_error(TypeError, "h1 should be of the same type {} of starting hamiltonian but "
                                   "is {}.".format(type(in_hamiltonian), type(fin_hamiltonian)))
        if in_hamiltonian.nqubits != fin_hamiltonian.nqubits:
            raise_error(ValueError, "H0 has {} qubits while final hamiltonian has {}."
                                    "".format(in_hamiltonian.nqubits, fin_hamiltonian.nqubits))
                
        self.circuit = circuit
        self._h0 = in_hamiltonian
        self._hf = fin_hamiltonian
        self._dt = dt
        self._T = total_time
                
        self._schedule = None
        self._param_schedule = None
        nparams = scheduling_func.__code__.co_argcount

        if not nparams == 1:
            raise_error(ValueError,"Scheduling function takes one argument,"
                                   "but {} were provided.".format(nparams))
        
        self._schedule = scheduling_func


   
    def set_SchedulingFunction(self, func):
        """ Set scheduling function as func. """
        ATOL = 1e-7 # Tolerance for boundary condittions.
        s0 = func(0)
        if np.abs(s0) > self.ATOL:
            raise_error(ValueError, "s(0) should be 0 but is {s0}.")
        s1 = func(1)
        if np.abs(s1 - 1) > self.ATOL:
            raise_error(ValueError, "s(1) should be 1 but is {s1}.")
        self._schedule = func

    def Hamiltonian(self, t):
        """
        Args:
            - t: time t so that s(real_t)=s(t/total_time)
        Returns hamiltonian at time t evolved as: H = ( 1 - s(t/T) ) h0 + s(t/T) hf
        """
        # boundary conditions  s(0)=0, s(total_time)=1
        st = self.SchedulingFunction(t)
        return self._h0 * (1 - st) + self._hf * st
    
    def SchedulingFunction(self, t):
        """
        Args
            t: time
        Returns scheduling function evaluated at time t s(t/total_time).
        """
        # returns value of scheduling function at time t
        return self._schedule(t / self._T)

    def SchedulingFunction(self):
        """ Returns scheduling function as a function of time. """
        if self._schedule is None:
            raise_error(ValueError, " Scheduling function is not defined.")
        return self._schedule
        
    def minimize(self, params, method="trust-constr", options=None):
        """
        Performs minimization to find the ground state
        Args:
            params (np.ndarray): Initial guess for the variational parameters.
            method (str): Name of optimizer to use. Default is scipy's trust-constr,
                            since from benchmarks it seems to be the most accurate one
                            for 4-8 qubits circuits.
            options (dictionary): additional settings.
        """
        
        # We first create a Hamiltonian suited for adiabatic evolution:
        # H = ( 1 - s(t) ) h0 + s(t) hf
        t=0
        H=self.Hamiltonian(t)
        
        # Prepare the ground state of H with s(0) = 0 using VQE.
        # Minimize it to find final parameters

        vqe = models.VQE(self.circuit, H)
        best0, finparams = vqe.minimize(params, method=method,
                                        options=options, compile=False)
    
        inparams=finparams
        s=self.SchedulingFunction(t)
        
        while s<=1:
            vqe = models.VQE(self.circuit, H)
            best, finparams = vqe.minimize(inparams, method=method,
                                            options=options, compile=False)
               
            # setup logger
            logging.info("Time "+ str(t)+ " terminated with value "+ str(best)+
                         "\n\t  scheduling function "+str(s) + "\nparams \n"+ str(finparams))

            # update
            t+=self._dt
            s=self.SchedulingFunction(t)
            H=self.Hamiltonian(t)
            inparams=finparams

        return best, finparams
