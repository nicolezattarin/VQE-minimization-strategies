import iminuit
import numpy as np
import qibo
from qibo import gates, models, hamiltonians
from deap import base, creator, tools, algorithms
from optuna import samplers, pruners, visualization
import optuna
import logging
import sys
import os
from spsa import *
from genetic_minimizer import*



def optimize(loss, InParams, args=(), method="migrad", options=None ):
    """
   Performs optimization of a loss function with method provided
    Args:
        loss (callable): Loss as a function
        InParams (np.ndarray): Initial guess for the variational
                                 parameters.
        args (tuple): optional arguments for the loss function.
        method (str): Name of optimizer to use.
                        Can be:
                            - migrad for iMinuit
                            - genetic for genetic algorithm
                            - hyperopt for optuna hyperoptimization alhorithm
                            - spsa for simultaneous perturbation stochastic approximation
       option (tuple): additional settings.

    """
    if method != "migrad" and method != "genetic" and\
       method !="cma" and  method !="bipop":
        raise NameError("{} is not a valid name for optimizer".format(method))
    
    if method == "migrad" :
        return iMinuitOptimizer(loss, InParams, args=args,
                       method=method, options=options)
                       
    elif method == "genetic":
        mygnetic = GeneticOptimizer (options)
        return mygnetic.minimize (loss, InParams, args)
        
    elif method == "bipop":
        return BipopCMAES (InParams, loss, args, options)

    elif method == "spsa":
        myspsa=SPSA(options)
        return myspsa.minimize(InParams, loss, args)
    else:
        return HyperoptOptimizer(InParams, loss, args,options)


def iMinuitOptimizer(loss, InParams, args=(), method="migrad", options=None):
    """
   Performs minimization using iMinuit minimizer
   Args:
        loss (callable): loss function to minimize
        InParams (np.ndarray): initial guess of parameters
        args (tuple): optional arguments for the loss function.
        method (str): method of minimization, migrad.
        options (tuple): special settings
    """
    result = iminuit.minimize(loss, InParams, args=args, method=method, options=options)
    """
    result is a dict with attribute access:
       x (ndarray): Solution of optimization.
       fun (float): Value of objective function at minimum.
       message (str): Description of cause of termination.
       hess_inv (ndarray): Inverse of Hesse matrix at minimum.
       nfev (int): Number of function evaluations.
       njev (int): Number of jacobian evaluations.
       minuit (object): Minuit object internally used to do the minimization.

    see https://iminuit.readthedocs.io/en/stable/reference.html?highlight=migrad#iminuit.minimize for details
    """
    print(result["message"],
          "\nNumber of function evaluations: ",result["nfev"])
    return result["fun"], result["x"]



def HyperoptOptimizer(inparams, loss, args=(), options={'nTrials': 1000,
                                                        'studyname': "hyperopt",
                                                        'verbose': True}):
    """
        Minimize a loss function
        Args:
            inparams (np.array): initial guess for parameters
            loss (callable): Loss as a function
            args (tuple):  arguments for the loss function,
            option (tuple): additional settings.
                                nTrials (int): number of trials for hyperoptimization
                                studyname (string): name for study, note that is the same used for db
                                storage, so we recommend not to use the same same for different tests.
                                verbose (bool): if true, a file with output informations is created.
    """
    
    def _objective(trial, nparams, loss, args):
        params=np.zeros(nparams)
        for i in range(nparams): #creates an array of dictionaries {"param":value}
            params[i]=trial.suggest_float("param"+str(i), 0, 2*np.pi)
        return loss(params, *args)
    
    nparams = len(inparams)
    studyname = options['studyname']
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(studyname)
        
    study = optuna.create_study(study_name=studyname,
                                storage = storage_name,load_if_exists=True)
                                
    study.optimize(lambda trial: _objective(trial, nparams, loss, args),
                                 n_trials=options['nTrials'], n_jobs=-1)
    params = study.best_params
    best = study.best_value
    
    #out file
    if options['verbose']:
        if (os.path.isdir(studyname+"_output")==False):
            os.mkdir(studyname+"_output")
        file=open(str(studyname)+"_output"+"/"+studyname+"_trials.txt", "w")
        tr = study.trials
        for i in range (len(tr)):
            file.write("trial "+str(i)+" value: " +str(tr[i].values)+\
                       " params: " + str(tr[i].params) +"\n")
    return best, params


def BipopCMAES (inparams, loss, args, options=None):
    """
    CMA-ES implmented with BIPOP option.
    BIPOP is a special restart strategy switching between two population sizings-small
    (like the default CMA, but with more focused search) and large (progressively increased as in IPOP).

    Args:
        loss (callable): Loss as a function
        args (tuple):  arguments for the loss function,
        option (dictionary): additional settings.

    """

    import cma
    r = cma.fmin2(loss, inparams, 1.7, options=options, bipop = True, args=args)
    return r[1].result.fbest, r[1].result.xbest
