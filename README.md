#### BCGP_ECI
BCGP_ECI(Bayesian Constrained Gaussian Proocess model for Extrapolations in CI methods) is the code used in the my paper, Phys. Rev. C 102, 024305 (2020), [arXiv:1907.04974v3].

The aim of the code is to provide extrapolations for full CI (and valence CI calculations in a future version) with uncertainty quantification.  

This is the minimal one to reproduce the values in the paper.  
Towards future possible applications, we will implement several things such as automatic selection of Kernel function/derivatives of Kernel for y' and y" evaluation/extension to higher dimension/etc.  

This code works at least in the following environment:  
macOS Catalina 10.15.3  
Julia v1.3.1, v1.4.2 
The author is not responsible for any type of troubles or damages caused by running the code.

#### Input format
<run.jl>: settings for Monte Carlo sampling etc.

<sample_ncsmdata.dat>: FCI results

The code now only supports the following format of FCI results:

[Nmax],[hbar omega],[g.s. Energy]

For the first time, execute $julia install_Pkgs.jl 
Then, execute $julia run.jl

### Outputs
1. Posterior*.dat:      #Posterior distributions for y^* (mean and stds are summarized)

2. Thetas*.dat:         #Hyperparameter distributions after "mstep" iteration of Metropolis-Hastings updates  
Tau(global strength), Sigma(correlation length), sample weight, ...


### References
S. Yoshida: "Non-parametric Bayesian approach to extrapolation problems in  configuration interaction methods", Phys. Rev. C 102, 024305 (2020).  
https://journals.aps.org/prc/abstract/10.1103/PhysRevC.102.024305

One can obtain the published FCI results from  
・(N3LO) M. K. G. Kruse, E. D. Jurgenson, P. Navrátil, B. R. Barrett, and W. E. Ormand, Phys. Rev. C 87, 044301 (2013).  
・(JISP16/NNLOopt) I. J. Shin, Y. Kim, P. Maris, J. P. Vary, C. Forssén, J. Ro- tureau, and N. Michel, Journal of Physics G: Nuclear and Particle Physics 44, 075103 (2017).  

MKL.jl(https://github.com/JuliaComputing/MKL.jl) improves the execution time.
