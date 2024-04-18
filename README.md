CHEN CRAFTS, my personal toolbox!
=================================

For now, there are three main parts in this package:
bsqubits, cqed and toolbox. 
They serves for different purposes in Danyang's research.


## Modules
- `Toolbox` (or `tb`): Toolbox includes functions for optimization, saving and loading data, etc. It is a general toolbox for all the projects.


- `cqed`: General codes for simulating the cqed systems. It includes simulations for pulse, decoherence, critical photon number, etc. I also define the FlexibleSweep class, inherited from the scqubits.ParameterSweep class, which helps to define swept parameters in a different way. 


- `bsqubits` (or `bsq`): A package for simulating and studying some spacific systems, especially for the resonator-qubit systems. The code isn't general enough to be used for other systems. Very high level and practical. Specifically, it has module 


- `projects` (or `prj`): A collection for all other projects, including files collected from other collaborators.

## Installation
```bash
pip install chencrafts
```