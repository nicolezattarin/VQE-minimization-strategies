import numpy as np
from qibo import models, callbacks, gates, hamiltonians
from qibo.abstractions import hamiltonians
import logging
from scipy import linalg
logging.basicConfig(level=logging.NOTSET)
import os



class SPSA (object):

    """
    Simultaneous Perturbation Stochastic Approximation.
    Args:
         eta (float): learning rate.
         eps (float): step size of expansion.
         maxiter (int): maximum number of iterations.
         etacorr (float): correction on learning rate, often reported as A.
         gamma (float): coefficient used to update eta.
         alpha (float): coefficient used to update eps.
        
            we do not recommend to change gamma and alpha

    """

    def __init__ (self, options={'eta':0.4,
                                 'eps': 1e-1,
                                 'maxiter':10000,
                                 'etacorrection':None,
                                 'alpha':0.101,
                                 'gamma': 0.602 ,
                                 'precision': 1e-10}):
  
        self.eta = options['eta']
        self.eps = options['eps']
        self.alpha  = options['alpha']
        self.gamma = options['gamma']
        self.maxiter = options['maxiter']
        self.precision = options['precision']
        
        if options['etacorrection']:
            self.etacorrection = options['etacorrection']
        else: self.etacorrection = 0.1 * self.maxiter

        
    def _calibration(self, params, loss, stat, args=()):
        """
        Calibrates eta coefficient before starting minimization.
        Args:
             - loss (Callable): loss function.
             - params (np.array): parameters at first iteration.
             - stat (int): number of iterations for calibration.
             - args (tuple): optional, arguments of loss function.
        """
    
        etaTarget = self.eta
        eps0 = self.eps
        deltaloss = 0
        for i in range(stat):
            delta = 2 * np.random.binomial (1,.5, len(params)) - 1
            paramsplus = params + eps0 * delta
            paramsmin = params - eps0 * delta

            lossplus = loss(paramsplus, *args)
            lossmin = loss(paramsmin, *args)
            deltaloss += np.abs(lossplus - lossplus) / stat
        # only calibrate if deltaloss is larger than 0
        if deltaloss > 0:
            self.eta = etaTarget * 2 / deltaloss \
                        * self.eps * (self.etacorrection + 1)


    def _gradient (self, params, loss, eps, args=()):
        """
        Calculate gradient for first-order SPSA
        gradf = ( f(r + eps * delta) - f(r - eps * delta) ) / (2 * eps)
         
        Args:
            params (np.array): parameters not perturbed.
            loss (callable): loss function.
            eps (float): update step.
            args (tuple): optional arguments for loss function.
        
        Return: gradient (np.array)
        """
        delta = 2 * np.random.binomial(1,.5, len(params)) - 1
        
        lossplus = loss(params + eps * delta, *args)
        lossmin = loss(params - eps * delta, *args)
        
        gradf = (0.5 /eps) * (lossplus - lossmin) * delta
        
        return gradf
        
                    
                    
    def minimize (self, params,
                        loss,
                        args = (),
                        calibration = True,
                        verbose = True):
        """
        Mimimizer to find the ground state of hamiltonian provided
        Args:
            params (np.array or list): initial guess for circuit parameters,
                                          may be chosen randomly.
            loss (callable): loss function to minimize.
            seed (int): seed for random samplings.
            precision (float): precision on stop condition due to
                                 parameters change.
            calibration (bool): if true, calibration of learning rate is done.
            verbose (bool): if true, steps of minimization are printed on terminal.
            
        Return:
            best (float): optimal value of loss function.
            params (np.array): parameters that realize the best.
        """
        # outfile
        if (os.path.isdir("outspsa")==False):
            os.mkdir("outspsa")
        minout=open("outspsa/minimization.txt", "w")
        minout.write("iteration\teps\teta\tbest\n")
        pout=open("outspsa/params.txt", "w")
        pout.write("iteration\tparams\n")
            
        k = 0
        oldparams = params*100
        np.random.seed(0)
        
        # calibration
        if calibration:
            nstep = min(25, max(1, self.maxiter // 5))
            self._calibration(params, loss, nstep, args)
        
        while np.linalg.norm(oldparams-params)/np.linalg.norm(oldparams)>self.precision\
              and k < self.maxiter:
            # The optimisation carried out until the solution has converged, or
            # the maximum number of itertions has been reached.
            
            eps = self.eps / ((k+1) ** self.alpha)
            eta = self.eta / ((k+1+self.etacorrection) ** self.gamma)
            
            oldparams = params
            gradf = self._gradient (params, loss, eps, args)

            # update
            params = oldparams - eta * gradf
            best = loss(params, *args)
            k += 1
            
            #minimization info
            if k % 100 == 0 and verbose:
                logging.info("Iteration "+ str(k-1)+ \
                            " terminated with value: "+ str(best)+\
                             "\nparams: \n"+ str(params) + "\n")
            #outfile
            minout.write(str(k)+"\t"+str(eps)+"\t"+str(eta)+"\t"+str(best)+"\n")
            pout.write(str(k)+"\t[")
            for i in range (len(params)): pout.write(str(params[i])+" ")
            pout.write(str(k)+"]")

        return best, params

