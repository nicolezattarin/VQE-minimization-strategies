#!/home/nicolezattarin/miniconda3/bin/python

import numpy as np
import qibo
from qibo import gates, models, hamiltonians
qibo.set_threads(1)
from ansatz import*
import pandas as pd
import argparse
import os, sys
from qutip import *
import matplotlib.pylab as plt
from matplotlib import cm
from qiskit.visualization import plot_state_qsphere
from sklearn import manifold

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, help="Number of qubits.", type=int)
parser.add_argument("--nlayers", default=4, help="Number of nlayers.", type=int)
parser.add_argument("--method", default="BFGS", help="Method of minimization.", type=str)
parser.add_argument("--dir", default="results", help="Directory to save data.", type=str)


def ParamsEvolution(inparams, loss, args, method, fileobj, options=None):
    """
    Performs minimization with callbacks
    """
    def callback(p):
        for i in range(len(p)):
            fileobj.write(str(p[i]))
            if i != ( len(p)-1 ):
                fileobj.write("\t")
        fileobj.write("\n")

    from scipy.optimize import minimize
    results = minimize(loss, inparams,args=args,
                        method=method, options=options, callback=callback)


    
    
def main(nqubits, nlayers, method, dir):
    """
    Creates a file with the evolution of parameters during minimization.
    Then this scripts finds the corresponding 'temporary' ground state of
    hamiltonian provided and plots it in a QSphere.
    
    Note that a QSphere is avaible up to 5 qubits.
    
    """
    def loss(params, hamiltonian, circuit):
    # returns the expectation value of hamiltonian in the final state.
        circuit.set_parameters(params)
        FinalState = circuit()
        return hamiltonian.expectation(FinalState).numpy()
    
    #setup for testing
    h = hamiltonians.XXZ(nqubits=nqubits)
    circuit = StandardCircuit(nqubits, nlayers)
    
    args=(h,circuit)
    np.random.seed(0)
    nparams = 2*nqubits*nlayers+nqubits
    inparams = np.random.uniform(0, 2*np.pi, nparams)
    
    #file of params
    if (os.path.isdir("results")==False):
        os.mkdir("results")
    file=open("results/"+str(nqubits)+"q_params_during_minimization_"+str(method)+".txt", "w")

    for i in range(nparams):
        file.write("param"+str(i))
        if i != ( nparams-1 ):
            file.write("\t")
    file.write("\n")

    
    #minimization
    options={'disp': True}
    ParamsEvolution(inparams, loss, args, method, file, options)
    file.close()
    
    #reading params
    params=pd.read_csv("results/"+str(nqubits)+\
                "q_params_during_minimization_"+str(method)+".txt",sep='\t')
    states=[]
    
    #find the state
    for i in range(len(params)):
        current_params = params.iloc[i,:]
        circuit.set_parameters(current_params.tolist())
        result = circuit.execute()
        states.append(list(result.state(numpy=True)))

    #save states
    file_s=open("results/"+str(nqubits)+"q_states_during_minimization_"\
            +str(method)+".txt", "w")
    for i in range(len(states)):
        file_s.write(str(states[i]))
        if i != ( len(states)-1 ):
                file_s.write("\t")
        file_s.write("\n")
    file_s.close()
    
    #plot qspheres
    if (os.path.isdir("qspheres")==False):
        os.mkdir("qspheres")
    i=0
    for state in states:
        fig,ax=plt.subplots(figsize=(10,10))
        plot_state_qsphere(state,ax=ax)
        fig.text(0.2,0.8, "Iteration "+ str(i), fontsize=30)
        fig.savefig("qspheres/"+str(nqubits)+"q_"+str(method)+"_qsphere"+str(i)+".png",
                    bbox_inches='tight')
        i+=1
        
        


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
