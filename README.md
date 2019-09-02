#### BCGP_ECI
BCGP_ECI(Bayesian Constrained Gaussian Proocess model for Extrapolations in CI methods) is the code used in the my paper, arXiv:1907.04974.

The aim of the code is to provide systematic and reasonable extrapolations for full CI and valence CI calculations.

### Version info.
**version 1 (for reproducibility of the results in the paper)

The v1 is the minimal one to reproduce the values in the paper.
Towards future possible applications, we will implement several things such as automatic selection of Kernel function/derivatives of Kernel for y' and y" evaluation/extension to higher dimension/etc.

This code works at least in the following environment:
macOS Sierra 10.12.6 & macOS High Sierra 10.13.6
Julia v1.0.3, Julia v1.1.0

The author is not responsible for any type of troubles or damages caused by running the code.

**v2 is opened(August 5th, 2019): "bayesCGP_v2.jl" and "submodule/BayesGPsubmodule_v2.jl"

One can now achieve calculations about 3 times faster than the previous version.


#### Input format
The sample input for MC sampling: <input.jl>

<sample_ncsmdata.dat>
The code now only supports the following format of FCI results.

[Nmax],[hbar omega],[g.s. Energy]

### How to run
In the first time, execute $julia install_Pkgs.jl 
1. Modify "input.jl"
2. $julia bayesCGP_vXX.jl

### Outputs
1. Posterior*.dat:       #Posterior distributions for y^* (mean and stds are summarized)

2. Thetas*.dat:         #Hyperparameter distributions after "mstep" iteration of Metropolis-Hastings updates

Weights, Tau(global strength), Sigma(correlation length)
