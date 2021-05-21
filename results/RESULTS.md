# VQE Benchmarks

We test minimization algorithm and approaches on the VQE problem. Here we present the benchmarks results.


Consider a variational circuit where every layer is made up by RY rotations followed by a layer of CZ gates in order to entangle the qubits, as shown in this figure: 

<img src="images/varlayer.png"  width="300" class="center"/>

We perform a VQE minimization, based on the previous circuit, using [qibo.models.VQE](https://qibo.readthedocs.io/en/stable/qibo.html#qibo.models.VQE.minimize) to find the ground state of a Heisenberg XXZ hamiltonian. 
Since it's possible to evaluate the minimum eigenvalue of an hamiltonian in Qibo, we can comprare the results of VQE minimization with the expected value.  So we will measure accuracy for different minimization algorithms, where we refer to accuracy as: log(1/eps), eps is the gap | result-expected |.

The configuration is repeated for a different number of layers and the input parameters are chosen randomly from 0 to 2pi in all benchmarks.

### Summary:
- Scipy and iMinuit optimizers
- CMA and genetic algorithm
- seed dependence 
- Hyperoptimization
- Adiabatically assisted VQE
- SPSA
- Training layer by layer
- SGD

## Scipy and iMinuit minimizers 

### Benchmarks results

We benchmark different minimization algorithm taken from [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) and migrad from [iminuit.minimize](https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.minimize). 


<img src="images/4q.png"  width="1000"/>  
<img src="images/6q.png"  width="1000" />
<img src="images/8q.png"  width="1000"/> 

It'ws clear that trust-constr, BFGS and L-BFGS-B are the most suitable algorithm for the given problem. COBYLA is very fast, but has the worst performances, expecially on circuits with highter number of qubits.

In general we observe that, increasing the number of qubits, accuracy decreases quickly. In addition to this, the plots show that after a certain number of layers, that depends on the number of qubits, accuracy doesn't increase more. 

Note: simulation performed on Galileo (1 thread)

### Evolution of the state
We can also investigate how parameters evolution through minimization looking at the evolution of the corresponding state on a QSphere. The final state is obviously the ground state of the hamiltonian that encodes our problem.

trust-constr (4 qubits, 4layers):

 <video controls="true" allowfullscreen="true"width="500">
    <source src="images/trust-constr_4q4l.mp4" type="video/mp4" >
    
Powell (4 qubits, 4layers):

 <video controls="true" allowfullscreen="true"width="500">
    <source src="images/powell_4q4l.mp4" type="video/mp4" >
    
CG (4 qubits, 4layers):

 <video controls="true" allowfullscreen="true"width="500">
    <source src="images/cg_4q4l.mp4" type="video/mp4" >

##CMA and genetic

### CMA
We test CMA optimizer implemented in Qibo on a 4 qubits circuit, first without extra options and then with a 1e-4 tolerance while the default one is 1e-11, in order to find the proper trade off between time and accuracy.
<img src="images/cma_simulation.png"  width="1000"/> 

<img src="images/cma_simulation_lowtol.png"  width="1000"/> 



We see that even setting a tolerance, that is 10e7 highter that the default one, time doesn't decrease significantly. While accuracy has the same issues in both cases: there are some circuit setups such that CMA doesn't converge.



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

##AAVQE 


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

