import numpy as np
from deap import base, creator, tools, algorithms
import logging
logging.basicConfig(level=logging.NOTSET)


class GeneticOptimizer():
    """
    Class for genetic optimizer.
    
    args: options (dictionary): contains all additional options to better control evolution
          PopSize: size of the population.
          MaxGen: maximum number of generations.
          cxpb: probability of mating two individuals.
          mutpb: probability of mutating an individual.
          mutindpb: probability of gene mutation.
    """
    
    def __init__ (self, options={'PopSize':150,
                                 'MaxGen':300,
                                 'cxpb':0.05,
                                 'mutpb':0.08,
                                 'mutindpb':0.4}):
                  
        self.PopSize = options ['PopSize']
        self.MaxGen = options ['MaxGen']
        self.cxpb = options ['cxpb']
        self.mutpb = options ['mutpb']
        self.mutindpb = options ['mutindpb']

    def _evolution(self, population,
                           toolbox,
                           loss,
                           args,
                           ngen,
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
            fitnesses.append(loss(invalid_ind[i], *args))
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
            offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind=[]
            for ind in population:
                if not ind.fitness.valid:
                    invalid_ind.append(ind)
            fitnesses = []
            for i in range (len(invalid_ind)):
                fitnesses.append(loss(invalid_ind[i], *args))
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
            
            if gen % 20 == 0:
                best = tools.selBest(population, k=1)
                bestValue = loss(best[0], *args)
                logging.info("Generation " + str(gen) +
                             " ended with best: " + str(bestValue) +
                             "\nparams: " + str(best[0]))


        return population
        

    def minimize(self, loss, InParams, args):
        """
        Optimization genetic algorithm:

        Args:
            - loss (callable): loss function to minimize
            - InParams (np.ndarray): initial guess of parameters
            - args (tuple): arguments for the loss function.
            - options (tuple): additional options, see class definition.
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
        toolbox.register("evaluate", loss, *args)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutindpb)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        POP_SIZE = self.PopSize
        NGEN = self.MaxGen
        CXPB = self.cxpb
        MUTPB = self.mutpb
        
        # set statistics, we're interested in the minimum
        stats = tools.Statistics()
        stats.register("min", min)

        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1)
        pop = self._evolution(pop, toolbox, loss, args, NGEN, stats, halloffame=hof)
        best = tools.selBest(pop, k=1)
        
        return loss(best[0], *args), best[0]


