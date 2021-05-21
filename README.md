
# VQE minimization strategies

**Introduction**

Quantum computers aim to solve problems with high computational costs, which are impossible for classical computers to approach. Nevertheless noise and decoherence are major problems on near term quantum devices. 
To overcome such issues Variational quantum algorithms are considered. These methods are quantum-classical hybrid optimization schemes that employ a classical minimizer to train a parametrized multiple-layer quantum circuit. 

## The problem 

My analysys focuses on the role played by optimizers during this procedure. I aim to find the family of minimizers that best suit different QML problems and to verify the role of entanglement. To do so I approach different strategies of minimization: quasi-Newton methods, heuristic techniques, hyperparameter optimization, stochastic approximation and adiabatic evolution. 

Therefore, with the code provided in this directory, I test different algorithms and strategies on the well-known problem of finding the ground state of an hamiltonian with Variational Quantum Eigensolver. I perform the simulation of quantum algorithms on classical devices with [Qibo](https://github.com/qiboteam/qibo), a framework for quantum simulation.

The results reached with this previous analysis are the basis to approach the problem of minimization in a Quantum Binary Classifier. Indeed I'm a collaborator of a project that aims to provide a quantum variational binary classifier (not public atm).

### Variational ansatz

Consider a variational circuit where every layer is made up by RY rotations followed by a layer of CZ gates in order to entangle the qubits, as shown in this figure: 

<img src="results/images/varlayer.png"  width="300" class="center"/>

We perform a VQE minimization, based on the previous circuit, using [qibo.models.VQE](https://qibo.readthedocs.io/en/stable/qibo.html#qibo.models.VQE.minimize) to find the ground state of a Heisenberg XXZ hamiltonian. 
Since it's possible to evaluate the minimum eigenvalue of an hamiltonian in Qibo, we can comprare the results of VQE minimization with the expected value.  So we will measure accuracy for different minimization algorithms, where we refer to accuracy as: log(1/eps), eps is the gap | result-expected |.

## Tested approaches

- **Scipy's algorithms:** 
- **IMinuit:** 
- **CMA:** 
- **Genetic algorithms:** 
- **Hyperoptimization as pure optimizer:** 
- **Adiabatically Assisted VQE:** 
- **Simultaneous perturbation stochastic approximation:** 
- **Training a single layer at a time:**
- **Stochastic gradient descent:**

## A taste of results

All the results are discussed in [RESULTS](https://github.com/nicolezatta/VQE-minimization-strategies/blob/main/results/RESULTS.md), but in order to arouse your interest I want to giva an overview of the possible results.

For instance the plot below provides a simple benchmark of Scipy's minimizers and Migrad on a 4 qubits. circuit:

<img src="results/images/4q.png"  width="1000"/>  

I'm also interested in studying the evolution of parameters (thus of states) during minimization, for example the QSphere below shows the evolution of the state toward the ground state for trust-constr algorithm:



<img src="results/images/trust-constr_4q_gif.gif"  width="400"/>  


