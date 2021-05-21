"""
Nicole Zattarin
Tuner for one-objective hyperoptimization
"""

#!/home/nicolezattarin/miniconda3/bin/python
import argparse
import time
import numpy as np
import qibo
from qibo import gates, models, hamiltonians
import sqlite3
import optuna
from optuna import visualization
qibo.set_threads(1)
from ansatz import*
import os, sys

sys.path.insert(0, os.path.abspath("../../optimizers"))
from spsa import*

"""
This code implements the tuning of hyperparameters eta and eps in SPSA algorithm.
The objective function provided is specific for VQE problem, but defining a different obj
the same main code can be used to tune other minimization problems.
Note that in this case obj returns the opposite of accuracy, in general objective function
should be defined as a one-objective function with a return value to minimize.

Two methods of parallelization are provided:
This implementations requires the installation of sqlite and parallelization is carried out as explained in:
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html

In addiction you can set manually the number of jobs in method study.optimize

Otherwise you can omit the creation of the database as explained in main, when storage_name is created
"""

parser = argparse.ArgumentParser()
parser.add_argument("--ntrials", default=1000, help="Number of trials for optuna optimization.", type=int)
parser.add_argument("--nqubits", default=4, help="Number of qubits.", type=int)
parser.add_argument("--nlayers", default=4, help="Number of layers.", type=int)




def obj (trial,nqubits, nlayers):
    # set the problem
    h=hamiltonians.XXZ(nqubits)
    circuit = StandardCircuit(nqubits, nlayers)

    args = (h, circuit)
    def loss(params, hamiltonian, circuit):
    # returns the expectation value of hamiltonian in the final state.
        circuit.set_parameters(params)
        FinalState = circuit()
        return hamiltonian.expectation(FinalState).numpy()

                
    #set scheduling function
    nparams = 2*nqubits*nlayers+nqubits
    params = np.random.uniform(0, 2*np.pi, nparams)
    
    eta = trial.suggest_float("eta", 1e-4,0.6)
    eps = trial.suggest_float("eps", 1e-6, 1e-1)
    
    options={'eta':eta,
                 'eps':eps,
                 'maxiter':10000,
                 'etacorrection':None,
                 'alpha':0.101,
                 'gamma': 0.602 ,
                 'precision': 1e-10}
    myspsa=SPSA(options=options)
    best, params = myspsa.minimize(params, loss, args, verbose = False)
    expected=np.real(np.min(h.eigenvalues().numpy()))
    
    return -np.log10(1/np.abs(best-expected))
    
   
def main(ntrials, nqubits, nlayers):

    np.random.seed(0)

    study_name = "spsatuning"+ str(nqubits)+"q" + str(nlayers) + "lay"
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    
    storage_name = "sqlite:///{}.db".format(study_name) # omitting this line and the arguments 'storage' and
                                                        #'load_if_exists' in optuna.create_study avoid the creation
                                                        #of a database and you can run the procedure without sqlite
    study = optuna.create_study(study_name=study_name,
                                storage = storage_name,  load_if_exists=True)
    study.optimize(lambda trial: obj(trial, nqubits, nlayers),
                                 n_trials=ntrials)#, n_jobs=-1) # suggested if using a cluster,
                                                                # in general it's possible to set
                                                                # manually the number of jobs
                                 
    # plots
    if (os.path.isdir("tuningspsa")==False):
        os.mkdir("tuningspsa")
    outstr = str(nlayers)+"lay"+str(nqubits)+"qub_"
    visualization.plot_contour(study).write_image("tuningspsa/"+outstr+"contour.png")
    visualization.plot_optimization_history(study).write_image("tuningspsa/"+outstr+"history.png")
    
    #out file
    file=open("tuningspsa/"+outstr+"_best_trials.txt", "w")
    best = study.best_trials
    for i in range (len(best)):
        file.write("best trial params: " + str(best[i].params) +\
                    " value: " +str(best[i].values)+ "\n")
    file=open("tuningspsa/"+outstr+"trials.txt", "w")
    tr = study.trials
    for i in range (len(tr)):
        file.write("best trial params: " + str(tr[i].params) +\
                    " value: " +str(tr[i].values)+ "\n")

    
if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
