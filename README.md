# Simulation for near-term quantum repeaters


[![DOI](https://zenodo.org/badge/573417154.svg)](https://zenodo.org/badge/latestdoi/573417154)


This repository is an archive for the code used in:

> ReQuSim: Faithfully simulating near-term quantum repeaters <br>
> J. Wallnöfer, F. Hahn, F. Wiesner, N. Walk, J. Eisert <br>
> Preprint: [arXiv:2212.03896 [quant-ph]](https://doi.org/10.48550/arXiv.2212.03896)

## Repository structure

The `scenarios` directory contains multiple different quantum repeater setups with files that allow to set up, run and evaluate their simulation.
As the core of the simulation some of the scenarios use an early, unreleased version of ReQuSim 
(consisting of the .py files in the repository root and the `libs` directory),
while others use v0.4 of the separately released [ReQuSim](https://github.com/jwallnoefer/requsim) Python Package &mdash; 
a circumstance which can only be described as _historically grown_ at the best of times.

The following scenarios are included:

| scenario name | description                                                                                                                                         |
|:-------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------|
|  twolink_epp  | Two repeater links with multiple memories and a protocol that can, optionally, perform a number of entanglement purification steps before swapping. |
|   manylink    | Variable number repeater links always using entanglement swapping as early as possible.                                                             |
| manylink_epp  | Variable number of repeater links with multiple memories and an adjustable number of entanglement purification steps at the lowest level.           |
|  many_params  | A variant of manylink_epp with an alternative error model (amplitude damping in memories and noisy two-qubit gates).                                |


Furthermore, there are also some scenarios that have known, analytical results for key rates. We used our simulation to re-obtain them numerically:

| scenario name | description                              |
| :-----------: | -----------------------------------------|
| luetkenhaus   | Two repeater links with a comprehensive noise model in [1] |
| multimemory   | The multi-memory variant of the above scenario as described in [2]. <br> Also includes a variant of that protocol but allowing for simultaneous entanglement generation. |
| whitepaper    | Two repeater links using parameters from experimental groups as described in the Q.Link.X "White Paper" [3] |


### How to use

If you wish to run the scenarios yourself, we recommend recreating the same virtual environment we used to develop the code via [pipenv](https://pipenv.pypa.io/en/latest/). 
This assumes a version of Python 3.8 is available on your system. 

```bash
pip install pipenv
pipenv sync
```

The scenario files are expected to be called from the repository root, e.g. to run an example configuration that outputs runtime and event stats use the following: 
```bash
pipenv run python scenarios/twolink_epp/two_link_epp.py
```

Some scenarios use only a single file to define the cases to be run and to directly run them. These scenario files begin with `run_` and are meant to be run on a cluster computer with multiple processors.
If that is not your usecase please adjust them accordingly.

Some of the scenarios use a `case_definiton.py` to define the cases to be run in a central location and a separate `run_` file to run each data point separately
(these offer a command line interface). The `orchestrate.py` files are a convenient way to create jobs on a HPC system that uses [Slurm](https://slurm.schedmd.com/) for scheduling 
&mdash; PLEASE change the email to receive notifications if you use these.

If you want to use the plot files use `pipenv sync --dev` instead (separate because some graphic libraries may not be available on HPC systems).

## Related projects

As mentioned above, the core part of the simulation has been released separately and is available as a Python package: 
[https://github.com/jwallnoefer/requsim](https://github.com/jwallnoefer/requsim), which is the variant of the simulation code that should be used going forward.

## References

[1] D. Luong, et al.; Appl. Phys. B __122__, 96 (2016); DOI: [10.1007/s00340-016-6373-4](https://doi.org/10.1007/s00340-016-6373-4) <br>
[2] R. Trényi, N. Lütkenhaus; Phys. Rev. A __101__, 012325 (2020); DOI: [10.1103/PhysRevA.101.012325](https://doi.org/10.1103/PhysRevA.101.012325) <br>
[3] P. van Loock, et al.; Adv. Quantum Technol. __3__, 1900141 (2020); DOI: [10.1002/qute.201900141](https://doi.org/10.1002/qute.201900141)
