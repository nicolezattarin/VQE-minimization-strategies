#!/home/nicolezattarin/miniconda3/bin/python
import optuna
import numpy as np
import qibo
from qibo import gates, models, hamiltonians
import logging
import time
from myvqe import *
from optuna import samplers, pruners, visualization
import argparse
from ansatz import*
import os
import sqlite3
import logging
import sys

qibo.set_threads(48)

parser = argparse.ArgumentParser()
parser.add_argument("--nTrials", default=1000, help="Number of trials for optuna optimization.", type=int)
parser.add_argument("--hyperopt", default="setup", help="hyperopt to perform, options are: 'setup', 'genetic'.", type=str)
parser.add_argument("--nqubits", default=4, help="Number of qubits.", type=int)
parser.add_argument("--nlayers", default=4, help="Number of layers.", type=int)
"""
We want to find the hyperparameters that minimize execution time and maximize accuracy of a VQE
minimizations where the ansatz is a variational circuit made up by multiple layers.

If hyperopt: 'setup' hyperparameters are:
                                        number of qubits
                                        number of layers
                                        minimization method from scipy.optimize.minimize
                                            
If hyperopt: 'genetic' hyperparameters are:
                                PopSize: population size
                                MaxGen: maximum number of generations
                                cxpb: crossover probability
                                mutpb: mutation probability
                                mutindpb:
"""



def objectiveGenetic(trial, nqubits, nlayers): # myVQE is an object from the class myVQE
    """
    Objective function for multi-objective optimization:
    we want to solve the trade off between time (to minimize) and accuracy (to maximize)
    in the genetic algorithm implemented in myminimizers.py
    To do so the hyperparameters are: PopSize, MaxGen, cxpb, mutpb
    Args:
         optuna.trial
         nqubits: number of qubits
         nlayers: number of layers
         
         (it's better if nlayer and nqubits are the best hyperparameters
         from objectiveSetup optimization)
    Returns: Time, Accuracy
    """
    
    PopSize = trial.suggest_int("PopSize", 50, 200)
    MaxGen = trial.suggest_int("MaxGen", 300, 500)
    cxpb = trial.suggest_float("cxpb", 0, 1)
    mutpb = trial.suggest_float("mutpb", 0, 1)
    mutindpb = trial.suggest_float("mutindpb", 0, 1)

    StartTime=time.time()
    circuit = StandardCircuit(nqubits, nlayers)
    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)
    
    # VQE minimization
    vqe = myVQE(circuit, hamiltonian)
    Expected = np.real(np.min(hamiltonian.eigenvalues().numpy()))
    
    np.random.seed(0)
    nParams = 2*nqubits*nlayers+nqubits
    InParameters = np.random.uniform(0, 2*np.pi, nParams)
    
    options = {'PopSize':PopSize, 'MaxGen':MaxGen, 'cxpb':cxpb ,
               'mutpb':mutpb, 'mutindpb':mutindpb}
    Best, Params = vqe.minimize(InParameters, method="genetic", options=options)
    Time = time.time()-StartTime
    Accuracy = np.log10(1/np.abs(Best-Expected))
    
    return Time, Accuracy

    
def objectiveSetup(trial):
    """
    Objective function for multi-objective optimization:
    we want to solve the trade off between time (to minimize) and accuracy (to maximize).
    To do so the hyperparameters are:
                                     number of qubits
                                     number of layers of the variational ansatz
                                     scipy minimizer
    Args: optuna.trial
    Returns: Time, Accuracy
    """
    nqubits = trial.suggest_int("nqubits", 2, 10, step=2)
    nlayers = trial.suggest_int("nlayers", 1, 10)
    method = trial.suggest_categorical("method",
                                        ["L-BFGS-B","Nelder-Mead",
                                        "Powell","BFGS","SLSQP",
                                        "trust-constr"])
    StartTime=time.time()
    # define the configuration for this specific problem
    circuit = StandardCircuit(nqubits, nlayers)
    hamiltonian=hamiltonians.XXZ(nqubits=nqubits)
    
    # VQE minimization
    VQE = models.VQE(circuit, hamiltonian)
    Expected = np.real(np.min(hamiltonian.eigenvalues().numpy()))
    
    np.random.seed(0)
    nParams = 2*nqubits*nlayers+nqubits
    InParameters = np.random.uniform(0, 2*np.pi, nParams)
    Best, Params = VQE.minimize(InParameters, method=method)
    Time = time.time()-StartTime
    Accuracy = np.log10(1/np.abs(Best-Expected))
    
    return Time, Accuracy
    


def main(nTrials, hyperopt, nqubits, nlayers):
    """
    Creates a file: "best_params.txt" containing the best setup for a VQE minimization with the variational
    ansatz (described in https://quantum-journal.org/papers/q-2020-05-28-272/) in order to maximize accuracy
    and minimize time.
    Provides pareto plot.
    Args:
         nTrials (int): number of trials.
         optimization (str): optimization to optimize (best setup, best parameters for genetic algorithm).
         nqubits (int): number of qubits.
         nlayers (int): number of layers.

    """

    
    if (hyperopt=="genetic"):
        study_name = "GenHyperparams"
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name,
                                    directions=["minimize", "maximize"],
                                    storage = storage_name,  load_if_exists=True,
                                    sampler=samplers.RandomSampler(),
                                    pruner=pruners.MedianPruner())
        study.optimize(lambda trial: objectiveGenetic(trial, nqubits, nlayers),
                                     n_trials=nTrials, n_jobs=-1)

    else:
        study_name = "VQEBestSetup"
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name,sampler=samplers.RandomSampler(),
                                    storage = storage_name, load_if_exists=True,
                                    pruner=pruners.MedianPruner(), directions=["minimize", "maximize"])
        study.optimize(objectiveSetup, n_trials=nTrials, n_jobs=-1)
        
    # random sampling, prune if the trial’s best intermediate result is worse than median of
    # intermediate results of previous trials at the same step.
    # From the benchmark results MedianPruner is the best for RandomSampler,
    # see https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako for details
    
    BestTrials = study.best_trials
    trials = study.get_trials
    
    #plots
    if (hyperopt == "genetic"): outstr=str(nqubits)+"qub_"+str(nlayers)+"lay_genetic"
    else: outstr="setup"
    if (os.path.isdir(hyperopt)==False):
        os.mkdir("../hyperopt")
    
    visualization.plot_pareto_front(study,
                  target_names=["time", "accuracy"], include_dominated_trials=True).write_image("../hyperopt/"+outstr+"_pareto.png")
    
    #out file
    file=open("../hyperopt/"+outstr+"_best_trials.txt", "w")
    file.write("BestTrials\n")
    for i in range(len(BestTrials)):
        file.write("best trial "+ str(i)+":\n")
        file.write("\t params: "+str(BestTrials[i].params)+"\n")
        file.write("\t values: "+str(BestTrials[i].values)+"\n")

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)

