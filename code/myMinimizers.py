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

def optimize(loss, InParams=[], args=(), method="migrad", options=None, nqubits=4, nlayers=4 ):
    """
   Performs optimization of a loss function with method provided
    Args:
        - loss (callable): Loss as a function
        - InParams (np.ndarray): Initial guess for the variational
                                 parameters.
        - args (tuple): optional arguments for the loss function.
        - method (str): Name of optimizer to use.
                        Can be:
                            - migrad for iMinuit
                            - genetic for genetic algorithm
                            - hyperopt for optuna hyperoptimization alhorithm
       - option (tuple): additional settings.
       - nqubits (int): number of qubits, necessary only if hyperopt optimizer is used
       - nlayers (int): number of layers, necessary only if hyperopt optimizer is used

    """

    if (method == "migrad" ):
        return iMinuitOptimizer(loss, InParams, args=args,
                       method=method, options=options)
    elif (method == "genetic"):
        return GAOptimizer (loss, InParams, args, options=options)
    else:
        return HyperoptOptimizer(loss, nqubits, nlayers, args, options)


def iMinuitOptimizer(loss, InParams, args=(), method="migrad", options=None):
    """
   Performs minimization using iMinuit minimizer
   Args:
        - loss (callable): loss function to minimize
        - InParams (np.ndarray): initial guess of parameters
        - args (tuple): optional arguments for the loss function.
        - method (str): method of minimization, migrad.
        - options (tuple): special settings
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


def GEvolution(population,
               toolbox,
               loss,
               args,
               cxpb=0.5,
               mutpb=0.5,
               ngen=300,
               stats=None,
               halloffame=None):
    """
    Reproduce an evolutionary algorithm:

    Args:
        - population (list): individuals.
        - toolbox: 'class deap.base.Toolbox' that contains the
                    evolution operators.
        - loss (callable): loss function
        - args (tuple): hamiltonian and circuit
        - cxpb (float): probability of mating two individuals.
        - mutpb (float): probability of mutating an individual.
        - ngen (int): number of generation.
        - halloffame: 'class deap.tools.HallOfFame' object that
                       will contain the best individuals, optional.
    Returns: Population after volution
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
    # Evaluate the individuals with an invalid fitness
    invalid_ind=[]
    for ind in population:
        if not ind.fitness.valid:
            invalid_ind.append(ind)
    fitnesses = []
    for i in range (len(invalid_ind)):
        fitnesses.append(loss(invalid_ind[i], args[0], args[1]))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)

    #hall of fame
    if halloffame is not None:
            halloffame.update(population)

    #set verbose and logbook
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    logbook.stream
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind=[]
        for ind in population:
            if not ind.fitness.valid:
                invalid_ind.append(ind)
        fitnesses = []
        for i in range (len(invalid_ind)):
            fitnesses.append(loss(invalid_ind[i], args[0], args[1]))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
    
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
    
        #set verbose and logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        logbook.stream

    return population
    

def GAOptimizer(loss, InParams, args, options={'PopSize':150,
                                               'MaxGen':300,
                                               'cxpb':0.05,
                                               'mutpb':0.08,
                                               'mutindpb':0.4}):
    """
    Optimization genetic algorithm:

    Args:
        - loss (callable): loss function to minimize
        - InParams (np.ndarray): initial guess of parameters
        - args (tuple): arguments for the loss function.
                        For VQE problem should be (hamiltonian, circuit)
        - options (tuple): (PopulationSize, MaxGenerations)
                           PopSize: size of the population.
                           MaxGen: maximum number of generation to consider.
    Returns: best, parameters
    """


    IND_SIZE = len(InParams) #Every individual is a set of parameters
                             #for the circuit
    np.random.seed(0)

    # Define individual's inherit class a list and the fitness attribute to
    # be the previously initialized Fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # initialization
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 2*np.pi)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                                   toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # operations
    toolbox.register("evaluate", loss, args[0], args[1])
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=options['mutindpb'])
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    POP_SIZE = options['PopSize']
    NGEN = options['MaxGen']
    CXPB = options['cxpb']
    MUTPB = options['mutpb']
    
    # set statistics, we're interested in the minimum
    stats = tools.Statistics()
    stats.register("min", min)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    pop = GEvolution(pop, toolbox, loss, args, CXPB, MUTPB, NGEN, stats, halloffame=hof)
    best = tools.selBest(pop, k=1)
    
    return loss(best[0], args[0], args[1]), best[0]


def HyperoptOptimizer(loss, nqubits, nlayers, args=(), options={'nTrials': 100}):
    """
        Minimize a loss function
        Args:
        - loss (callable): Loss as a function
        - nqubits (int): number of qubits
        - nlayers (int): number of layers
        - args (tuple):  arguments for the loss function,
                         For VQE problem should be (hamiltonian, circuit)
        - option (tuple): additional settings.
    """
    
    def _objective( trial, hamiltonian, circuit, nParams, loss):
        params=np.zeros(nParams)
        for i in range(nParams): #creates an array of dictionaries {"param":value}
            params[i]=trial.suggest_float("param"+str(i), 0, 2*np.pi)

        return loss(params, hamiltonian, circuit)
        
    nParams = 2*nqubits*nlayers+nqubits
        
    study_name = "hopt_"+str(nqubits)+"qub_"+str(nlayers)+"lay"
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
        
    study = optuna.create_study(study_name=study_name,
                                storage = storage_name,load_if_exists=True)
                                
    study.optimize(lambda trial: _objective(trial, args[0], args[1], nParams, loss),
                                 n_trials=options['nTrials'], n_jobs=-1)
    bestParams = study.best_params
    best = study.best_value
        
    return best, bestParams
