#### BCGP_ECI
BCGP_ECI(Bayesian Constrained Gaussian Proocess model for Extrapolations in CI methods) is the code used in the my paper, arXiv:XXXX.

The aim of the code is to provide systematic and reasonable extrapolations for full CI and valence CI calculations.

The current version v1 (July, 2019) is the minimal one to reproduce the values in the paper.
Towards future possible applications, we will implement several things such as automatic selection of Kernel function/derivatives of Kernel for y' and y" evaluation/extension to higher dimension/etc.

#### Input format
The sample file is input.jl

<input.jl>
numN=2000               # Number of particle for MC sampling. numN > 20,000 is recommended!
mstep=2000              # Number of iteration of main step
Rstep=500               # Number of iteration to calc. R_E
Kernel="logMatern"      # Kernel function "Lin+Mat"/"RQ"/"RBF"/"Matern"/"Mat32"/"logRBF"/"logMat32"/"NSMatern"/"logRQ"
sigfac=1                # R_E = Rmean + sigfac * Rstd
Auxiliary=false         # Not used
Monotonic=false         # Not used
Convex=false            # Not used
fixed=false             # To keep hyperparameters fixed (debug option)
printder=false          # Not used
multihw=false           # Not used
rMH=[0.30,0.30]         # desired acceptance ratio for adaptive proposals
CImode ="NCSM"          # "MCSM" mode is now under construction

nuc="6Li"               # target nuclus (Not used: needed only for my own module to plot)
inpname="sample_ncsmdata.dat"    # input file
qT=1.e-1                # scale factor of proposal for hyperparameters
qY=1.e-3                # scale factor of proposal for predictions y*
qYfac = 5.e-3           # reduction factor of qY after "Resampling"
xpMax=46                # Maximum Nmax to be calculated. Start with large number and then decrease if you don't need the large number.

<sample_ncsmdata.dat>
The code now only support the following format of FCI results.

[Nmax],[hbar omega],[g.s. Energy]


### How to run
1. Modify "input.jl"
2. $julia bayesCGP_v1.jl

### Outputs
1. Posterior*dat:       #Posterior distributions for y* (mean and stds are summarized)
2. Thetas*.dat:         #Hyperparameter distributions after mstep iteration of Metropolis-Hastings updates
