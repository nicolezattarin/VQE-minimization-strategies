"""
Nicole Zattarin
Optimizers
"""

import iminuit
import numpy as np
import qibo
from qibo import gates, models, hamiltonians
from deap import base, creator, tools, algorithms
from optuna import samplers, pruners, visualization
import optuna
import nlopt
import logging
import sys,os
from spsa import *
from genetic import*
from scipy.optimize import Bounds
import warnings


def optimize(loss, InParams, args=(), method="migrad", options=None, callback=None):
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
                            - bipop
                            - ISRES
                            - AGS
                            - PSO
       option (tuple): additional settings.
       callback (callable): callback for optimizers.
       

    """
    method=method.lower()
    
    if method != "migrad" and method != "genetic" and\
       method !="spsa" and  method !="bipop" and \
       method !="hyperopt" and method !="isres" and\
       method != "ags" and method !="pso" and method !="shgo":
        raise RuntimeError(" {} is not a valid name for optimizer".format(method))
    
    if method == "migrad" :
        return iMinuitOptimizer(loss, InParams, args=args,
                       method=method, options=options, callback=callback)
                       
    elif method == "genetic":
        #set options
        if options!=None:
            mygnetic = GeneticOptimizer (options)
        else:
            mygnetic = GeneticOptimizer ()

        return mygnetic.minimize (loss, InParams, args, callback=callback)
        
    elif method == "bipop":
        if callback:
            warning.warn("Callbacks are not supported for {}".format(method), RuntimeWarning)
        return BipopCMAES (InParams, loss, args, options)

    elif method == "spsa":
        myspsa=SPSA(options)
        return myspsa.minimize(InParams, loss, args, callback=callback)
        
    elif method == "isres":
        if callback:
            warning.warn("Callbacks are not supported for {}".format(method), RuntimeWarning)
        return ISRES (InParams, loss, args, options)
        
    elif method == "ags":
        if callback:
            warning.warn("Callbacks are not supported for {}".format(method), RuntimeWarning)
        return AGS (InParams, loss, args, options)
        
    elif method == "pso":
        if callback:
            warning.warn("Callbacks are not supported for {}".format(method), RuntimeWarning)
        return PSO (InParams, loss, args, options)
    
    elif method == "shgo":
        if callback:
            warning.warn("Callbacks are not supported for {}".format(method), RuntimeWarning)
        return SHGO (InParams, loss, args, options)
    else:
        if callback:
            warning.warn("Callbacks are not supported for {}".format(method), RuntimeWarning)
        return HyperoptOptimizer(InParams, loss, args,options)


def iMinuitOptimizer(loss, InParams, args=(), method="migrad", options={}, callback=None):
    """
   Performs minimization using iMinuit minimizer
   Args:
        loss (callable): loss function to minimize
        InParams (np.ndarray): initial guess of parameters
        args (tuple): optional arguments for the loss function.
        method (str): method of minimization, migrad.
        options (tuple): special settings
    """
    result = iminuit.minimize(loss, InParams, args=args, method=method, options=options, callback=callback)
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



def HyperoptOptimizer(inparams, loss, args=(), options=None):
    """
        Minimize a loss function with hyperoptimization approach.
        Args:
            inparams (np.array): initial guess for parameters
            loss (callable): Loss as a function.
            args (tuple):  arguments for the loss function,
            option (tuple): additional settings.
                                nTrials (int): number of trials for hyperoptimization.
                                studyname (string): name for study, note that is the same used for db.
                                storage, so we recommend not to use the same same for different tests.
                                verbose (bool): if true, a file with output informations is created.
    """
    default_options = {'nTrials': 1000,
                       'studyname': "hyperopt",
                       'verbose': False}
    #set options
    if options!=None:
        options={**default_options, **options}
    else:
        options=default_options
    
    def _objective(trial, nparams, loss, args):
        params=np.zeros(nparams)
        for i in range(nparams):
            params[i]=trial.suggest_float("param"+str(i), 0, 2*np.pi)
        return loss(params, *args)
    
    
    """
    Two methods of parallelization are provided:
    This implementations requires the installation of sqlite and parallelization is carried out as explained in:
    https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
    
    In addiction you can set manually the number of jobs in method study.optimize

    Otherwise you can omit the creation of the database and just set the number of jobs manually
    """
    nparams = len(inparams)
    studyname = options['studyname']
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(studyname)
        
    study = optuna.create_study(study_name=studyname,
                                storage = storage_name,load_if_exists=True)
                                
    study.optimize(lambda trial: _objective(trial, nparams, loss, args),
                                 n_trials=options['nTrials'])#, n_jobs=-1)
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

    return best, list(params.values())


def BipopCMAES (inparams, loss, args=(), options={}):
    """
    CMA-ES implmented with BIPOP option.
    BIPOP is a special restart strategy switching between two population sizings-small
    (like the default CMA, but with more focused search) and large (progressively increased as in IPOP).

    Args:
        inparams (np.array): initial guess for parameters.
        loss (callable): Loss as a function.
        args (tuple):  arguments for the loss function.
        option (dictionary): additional settings.

    """

    import cma
    r = cma.fmin2(loss, inparams, 1.7, options=options, bipop = True, args=args)
    return r[1].result.fbest, r[1].result.xbest


def ISRES (inparams, loss, args=(), options=None):
    """
    "Improved Stochastic Ranking Evolution Strategy" (ISRES)
    algorithm for nonlinearly-constrained global optimization
    
    Args:
        inparams (np.array): initial guess for parameters.
        loss (callable): Loss as a function.
        args (tuple):  arguments for the loss function.
        option (dictionary): additional settings.
    """

    default_options = {'tol': 1e-8,
                       'maxeval': 4000,
                       'lower_bound': 0,
                       'upper_bound': 2*np.pi}
                       
    #set options
    if options!=None:
        options={**default_options, **options}
    else:
        options=default_options
        
    opt = nlopt.opt(nlopt.GN_ISRES, len(inparams))
    opt.set_min_objective(lambda x, grad: loss(x, *args))
    
    #setting options
    opt.set_lower_bounds(options['lower_bound'] * np.ones(len(inparams)))
    opt.set_upper_bounds(options['upper_bound'] * np.ones(len(inparams)))
    
    opt.set_ftol_rel(options['tol'])
    opt.set_maxeval(options['maxeval'])
    
    #minimization
    params = opt.optimize(inparams)
    best = loss(params, *args)
    
    return best, params
    
def AGS (inparams, loss, args=(), options=None): #to fix
    """
    An implementation of the algorithm AGS to solve constrained
    nonlinear programming problems with Lipschitzian functions.
    
    Args:
        inparams (np.array): initial guess for parameters.
        loss (callable): Loss as a function.
        args (tuple):  arguments for the loss function.
        option (dictionary): additional settings.
    """

    default_options = {'tol': 1e-8,
                       'maxeval': 4000,
                       'lower_bound': 0,
                       'upper_bound': 2*np.pi}
                       
    #set options
    if options!=None:
        options={**default_options, **options}
    else:
        options=default_options
        
    opt = nlopt.opt(nlopt.GN_AGS, len(inparams))
    opt.set_min_objective(lambda x, grad: loss(x, *args))
    
    #setting options
    opt.set_lower_bounds(options['lower_bound'] * np.ones(len(inparams)))
    opt.set_upper_bounds(options['upper_bound'] * np.ones(len(inparams)))
    
    opt.set_ftol_rel(options['tol'])
    opt.set_maxeval(options['maxeval'])
    
    params = opt.optimize(inparams)
    best = loss(params, *args)

    return best, params


def PSO (inparams, loss, args, options=None):
    from pso import PSO

    """
    Particle swarm optimization
    
    Args:
        inparams (np.array): initial guess for parameters.
        loss (callable): Loss as a function.
        args (tuple):  arguments for the loss function.
        option (dictionary): additional settings.
    """

    default_options = {'bounds': [(0., 2*np.pi) for _ in range(len(inparams))],
                       'num_particles': 15,
                       'maxiter': 500}
                       
    #set options
    if options!=None:
        options={**default_options, **options}
    else:
        options=default_options
        
    mypso = PSO(lambda x: loss(x, *args), inparams,
                bounds=options['bounds'],
                num_particles=options['num_particles'],
                maxiter=options['maxiter'])
    params = mypso.out()
    best = loss(params, *args)
    
    return best, params
    
def SHGO (inparams, loss, args=(), options={}):
    from scipy.optimize import shgo

    """
    Simplicial homology global optimization
    
    Args:
        inparams (np.array): initial guess for parameters.
        loss (callable): Loss as a function.
        args (tuple):  arguments for the loss function.
        option (dictionary): additional settings.
        bounds (np.array or list): bounds for constrained optimization.
    """
    
    default_options={'f_min': 0.09,
                     'disp': True,
                     'maxfevint': 500,
                     'maxev': 20000}
             
    #set options
    if options!=None:
        options={**default_options, **options}
    else:
        options=default_options
        
    bounds = [(0., 2*np.pi) for _ in range(len(inparams))]
            
    r = shgo(loss, bounds=bounds, options=options,
            args=args,
            minimizer_kwargs={"ftol": 1e-6})

    print(r.message)
    print("Number function evaluation: ", r.nfev)
    
    return r.fun, r.x

