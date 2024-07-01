CHENCRAFTS, Danyang's personal toolbox!
=================================

There are four main parts in this package: `Toolbox`, `cqed`, `bsqubits`, and `projects`. They serves for different purposes in Danyang's research.


## Modules
- `Toolbox` (or `tb`): Toolbox includes functions for optimization, saving and loading data, etc. It is a general toolbox for all the projects.

- `cqed`: General codes for simulating the cqed systems. It includes simulations for pulse, decoherence, critical photon number, etc. I also define the FlexibleSweep class, inherited from the scqubits.ParameterSweep class, which helps to define swept parameters flexibly. Specifically, it has module `custom_sweeps` for a bunch of pre-defined custom sweeps, which can be used in `scqubits.ParameterSweep` class.

- `bsqubits` (or `bsq`): A package for simulating and studying some spacific systems, especially for the resonator-qubit systems. The code isn't general enough to be used for other systems. Very high level and practical. Specifically, it has module `QEC_graph` for simulating the cat code using a graph representation.

- `projects` (or `prj`): A collection for all other projects, including files collected from other collaborators.

- `fluxonium` (or `fx`): Like `bsqubits`, it's also a code collection for a specific project. It has codes to calculate gate fidelity for FRF system and perform the corresponding analysis.

## Installation
```bash
pip install chencrafts
```
