# VQE Benchmarks
We test minimization algorithm and approaches on the VQE problem. Here we present the benchmarks results.

Consider a variational circuit where every layer is made up by RY rotations followed by a layer of CZ gates in order to entangle the qubits, as shown in this figure: 

<img src="images/varlayer.png"  width="300" class="center"/>

This same circuit has been used as ansatz for a variational quantum algorithm implemented to benchmark the accuracy of the Variational Quantum Eigensolver (VQE) based on a finite-depth variational quantum circuit encoding ground states of local Hamiltonians, namely, the Ising and XXZ models. See [Scaling of variational quantum circuit depth for condensed matter systems](https://quantum-journal.org/papers/q-2020-05-28-272/).

We perform a VQE minimization, based on the previous circuit, using [qibo.models.VQE](https://qibo.readthedocs.io/en/stable/qibo.html#qibo.models.VQE.minimize) to find the ground state of a Heisenberg XXZ hamiltonian. 
Since it's possible to evaluate the minimum eigenvalue of an hamiltonian in Qibo, we can compare the results of VQE minimization with the expected value.  So we measure accuracy as: log(1/eps), and eps is the gap between the results achieved through minimization and the real ground state eigenvalue.

The configuration is repeated for a different number of layers and the input parameters are chosen randomly from 0 to 2pi in all benchmarks.

### Summary:
- Scipy and iMinuit optimizers
- CMA and genetic algorithm
- Seed dependence and stability
- Hyperoptimization
- Adiabatically assisted VQE
- SPSA
- Training layer by layer
- SGD

## Scipy and iMinuit minimizers 

### Algorithms

We benchmark different minimization algorithms taken from [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) and migrad from [iminuit.minimize](https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.minimize). 
The algorithms of this famiy are based on different strategies, in detail:
 
- **Unconstrained minimization**
	- Nelder-Mead: based on the [Simplex algorithm ](https://academic.oup.com/comjnl/article-abstract/7/4/308/354237));

	- BFGS: a quasi-Newton method, it uses the first derivatives only. It is known that this algorithm behaves well also with non-smooth optimizations;

- **Bound-Constrained minimization**

	- L-BFGS-B: a bound constrained minimizer based on: [Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization](https://dl.acm.org/doi/10.1145/279232.279236), and [A Limited Memory Algorithm for Bound Constrained Optimization](https://epubs.siam.org/doi/abs/10.1137/0916069?journalCode=sjoce3&mobileUi=0);

	- Powell: the one implemented in Scipy is a variation to the classical Powell method, for which function need not be differentiable, and no derivatives are taken. If bounds are not provided, as in this case, then an unbounded line search will be used. It performs sequential one-dimensional minimizations along each direction, which is updated at each iteration of the main minimization loop. 

	- TNC: a gradient-based truncated Newton algorithm to minimize a function with variables subject to bounds. 

- **Constrained Minimization**

	- COBYLA: it performs the Constrained Optimization By Linear Approximation (COBYLA) method. The algorithm is based on linear approximations to the objective function and each constraint;

	- SLSQP: uses Sequential Least SQuares Programming to minimize a function of several variables with any combination of bounds, equality and inequality constraints;
	- trust-constr: is a trust-region algorithm for constrained optimization.It is presented in Scipy as the most versatile constrained minimization algorithm implemented in the module and the most appropriate for large-scale problems. 




### Results

<img src="images/4q.png"  width="1000"/>  
<img src="images/6q.png"  width="1000" />
<img src="images/8q.png"  width="1000"/> 

It's clear that trust-constr, BFGS and L-BFGS-B are the most suitable algorithms for the given problem. COBYLA is very fast, but has the worst performances, expecially on circuits with high number of qubits.

In general we observe that, increasing the number of qubits, accuracy decreases quickly. In addition to this, the plots show that after a certain number of layers, that depends on the number of qubits, the new layers added don't affect the performances of the circuit. This means that our architecture is flexible with a few layers, while when parameters increase 

Note: simulation performed on LCM cluster (1 thread)

### Evolution of the state
We can also investigate how parameters evolve through minimization, looking at the evolution of the corresponding state on a QSphere. The final state is obviously the ground state of the hamiltonian that encodes our problem.
The QSpheres below show the ground state that correspond to every step of minimization for different algorithms taken from Scipy.

For trust-constr (4 qubits, 4layers):
<p align="center">
	<img src="images/trust-constr_4q_gif.gif"  width="400"/>  
</p>
     
For Powell (4 qubits, 4layers):
<p align="center">
	<img src="images/powell_4q_gif.gif"  width="400"/>  
</p>

    
For CG (4 qubits, 4layers):
<p align="center">
	<img src="images/cg_4q_gif.gif"  width="400"/>  
</p>

## CMA and genetic

### CMA
We test CMA optimizer on a 4 qubits circuit. CMA-ES stands for Covariance matrix adaptation evolution strategy (CMA-ES)  

The tests are performed first without extra options and then with a 1e-4 tolerance (the default one is 1e-11), in order to find the proper trade off between time and accuracy. Indeed GAs typically take more time than grandient based algorithms, because they are based on a heuristic search in space of possible solutions. 
<img src="images/cma_simulation.png"  width="1000"/> 

<img src="images/cma_simulation_lowtol.png"  width="1000"/> 



We see that even setting a tolerance, that is 10e7 higher that the default one, time doesn't decrease significantly. While accuracy has the same issues in both cases: there are some circuit setups such that CMA doesn't converge.



### CMA evolution
We now study CMA's function evaluations and best value ever through minimization. For different layouts of the the circuit we obtain these plots:

<img src="images/2lay4qub.png"  width="800"/> 
<img src="images/4lay4qub.png"  width="800"/>
<img src="images/6lay4qub.png"  width="800"/>
<img src="images/2lay6qub.png"  width="800"/>

We see that accuracy is highter in those cases wherethe algorithm finds it's way to the minimum earlier, without oscillating around a fixed value. Anyway we can't be confident using cma, since there are some circuits's layout for which it doesn't converge.
Note that running the same algorithm different times may produce a bit different output, because of the  evolutionary approach.

Note: runned locally.

### Bipop
We now test CMA with BIPOP option.   BIPOP is a special restart strategy switching between two population sizings-small
    (like the default CMA, but with more focused search) and large (progressively increased as in IPOP).

The results are shown in the plots below:
<img src="images/bipop.png"  width="1000"/>

As previously, we study CMA's approach to the best. For certain layouts we observe this kind of evolution:

<img src="images/4lay4qub_bipop.png"  width="800"/>
<img src="images/6lay4qub_bipop.png"  width="800"/>

We can conclude that even with the BIPOP mechanism, convergence is not guaranteed. 

Note: runned on Galileo
### Genetic

We then implement a genetic algorithm with [deap](https://deap.readthedocs.io/en/master/), tuning parameters with [Optuna](https://optuna.org). 
The evolution strategy consists of a two points crossover, selction of the best individual among tournsize randomly chosen individuals and a mutation that flips the value of the attributes of the input individual.

We notice that, even tuning parameters, the algorithm can't reach convergence, as shown by these plots:
<img src="images/genetic.png"  width="4000"/>



 Note: runned on Galileo.
 
 
## Error due to different seeds

The initial parameters of the optimization are stored in an array of random values extracted uniformly in (0, 2pi). We wand to study if there is a dependency of convergence on the seed. Repeating the same simulation with different seeds we observe the oscillations shown in these plots:

<img src="images/4qtrust-constr_error.png"  width="500"/>  <img src="images/4qBFGS_error.png"  width="500"/> 

<img src="images/4qL-BFGS-B_error.png"  width="500"/> <img src="images/4qNelder-Mead_error.png"  width="500"/>

<img src="images/4qSLSQP_error.png"  width="500"/>  <img src="images/4qmigrad_error.png"  width="500"/>  

 <img src="images/4qPowell_error.png"  width="500"/>  <img src="images/4qCOBYLA_error.png"  width="500"/>
  
 <img src="images/4qcma_error.png"  width="500"/>
 
It's possible to observe that trust-constr, that is the most accurate algorithm tested for a 4 qubits circuit, is also the less affected by seed change. On the other hand, COBYLA and Powell, that have low performances, depends on seed. Notice also that CMA convergence has a clear dependency on seed.


## Hyperoptimization as pure optimizer
We then implemented an optimizer based on pure hyproptimization with Optuna. With
7000 trials we get this plot:

 <img src="images/hyperopt.png"  width="600"/>

It's clear that this method doesn't converge to the right solution. In general increasing the number of trials should bring to better performance, because of the aleatory nature of the algorithm. But notice that, after thousand of trials, the best trial ever becomes almost constant.

## AAVQE 


We now perform an Adiabatically Assisted VQE as explained in [Addressing hard classical problems with Adiabatically Assisted Variational Quantum Eigensolvers](https://arxiv.org/abs/1806.02287v1)
with a linear scheduling function.
We repeat the simulation with different minimization algorithms for VQE optimization during adiabatic discrete evolution.



<img src="images/4qubitsAAVQE.png"  width="4000"/>
<img src="images/6qubitsAAVQE.png"  width="4000"/>
Note: performed on Galileo, 1cpu per task, 1 thread set in qibo)




to do: tune different scheduling functions

## SPSA

We now implement first order spsa as explained in [1704.05018v2](https://arxiv.org/pdf/1704.05018v2.pdf#section*.11). with learning rate calibration as it's done in [qiskit.aqua.components.optimizers.SPSA](https://qiskit.org/documentation/_modules/qiskit/aqua/components/optimizers/spsa.html#SPSA).

Before testing the algorithm on VQE problem we tune the step eps and the learning rate eta in order to maximize accuracy. Here we report the contour plots of tuning, for different number of qubits and a 4 layer circuit:


<img src="images/4lay2qub_contour.png"  width="400"/> <img src="images/4lay4qub_contour.png"  width="400"/>

<img src="images/4lay6qub_contour.png"  width="400"/><img src="images/4lay8qub_contour.png"  width="400"/>

According to these plots we decide to set eta = 0.4 and eps =1e-1 as default parameters, they can be changed according to different loss functions.

For our specific problem we set the parameters that we got from tuning. Note that the algorithm has hight dependency on the number of qubits. 


<img src="images/spsa_different_qubits.png"  width="1000"/>


## Training single layer
As last approach we optimized single-layers, fixing the rest of the trainable elements of the ansatz. We repeated these single-parameter and single-layer optimization cycles until we reached convergence. 
Loop stops when:


<img src="images/precision_formula.png"  width="250"/>
	
Precision default value is 1e-10. It can be taken smaller, but sometimes this choice makes the convergence impossible to reach, since with small circuits the parameters change too much at each iteration. So we recommend to choose higher precision if the circuit has less than 10/12 parameters and the algorithm used to minimize is not stable. For instance we faced this problem with COBYLA and a 4 qubits, one layer circuit. 

<img src="images/4q_single_layer.png"  width="1000"/>

## SGD 
For sake of completeness we also tested the [tf.keras.optimizers] (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
<img src="images/sgd4q.png"  width="1000"/> <img src="images/sgd6q.png"  width="1000"/>
<img src="images/sgd8q.png"  width="1000"/>

