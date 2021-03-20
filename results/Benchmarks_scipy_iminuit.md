# Benchmarks
## Scipy and iMinuit minimizers 

Consider a variational circuit where every layer is made up by RY rotations followed by a layer of CZ gates in order to entangle the qubits, as shown in this figure: 

<img src="varlayer.png"  width="300" class="center"/>

We perform a VQE minimization, based on the previous circuit, using [qibo.models.VQE](https://qibo.readthedocs.io/en/stable/qibo.html#qibo.models.VQE.minimize) in order to find the ground state of a Heisenberg XXZ hamiltonian. 
Since it's possible to evaluate the minimum eigenvalue of an hamiltonian in Qibo, we can comprare the results of the VQE minimization with the expected value. 

We benchmarked different minimization algorithm taken from [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) and migrad from [iminuit.minimize](https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.minimize). Take as measure of the accuracy of VQE with different minimization algorithms: log(1/eps), where eps is the gap |result-expected|.


The configuration is repeated for a different number of layers and the input parameters are chosen randomly from 0 to 2pi in all benchmarks.

Simulation with 4 qubits circuit:

<img src="4q.png"  width="700" class="center"/> 


Simulation with 6 qubits circuit:

<img src="6q.png"  width="700" class="center"/>


Simulation with 8 qubits circuit:

<img src="8q.png"  width="700" class="center"/> 


Accuracy is plotted with the same limits on y axis in order to better compare different number of qubits circuits. 
Note: simulation performed on Galileo (1 cpu per task)