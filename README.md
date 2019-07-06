#### BCGP_ECI
BCGP_ECI(Bayesian Constrained Gaussian Proocess model for Extrapolations in CI methods) is the code used in the my paper, arXiv:XXXX.

The aim of the code is to provide systematic and reasonable extrapolations for full CI and valence CI calculations.

The current version v1 (July, 2019) is the minimal one to reproduce the values in the paper.
Towards future possible applications, we will implement several things such as automatic selection of Kernel function/derivatives of Kernel for y' and y" evaluation/extension to higher dimension/etc.

#### Input format
The sample input files:

<input.jl>
L1:numN=2000               # Number of particle for MC sampling. numN > 20,000 is recommended!
L2:mstep=2000              # Number of iteration of main step
L3:Rstep=500               # Number of iteration to calc. R_E
L4:Kernel="logMatern"      # Kernel function "Lin+Mat"/"RQ"/"RBF"/"Matern"/"Mat32"/"logRBF"/"logMat32"/"NSMatern"/"logRQ"
L5:sigfac=1                # R_E = Rmean + sigfac * Rstd
L6:Auxiliary=false         # Not used
L7:Monotonic=false         # Not used
L8:Convex=false            # Not used
L9:fixed=false             # To keep hyperparameters fixed (debug option)
L10:printder=false          # Not used
L11:multihw=false           # Not used
L12:rMH=[0.30,0.30]         # desired acceptance ratio for adaptive proposals
L13:CImode ="NCSM"          # "MCSM" mode is now under construction

L15:nuc="6Li"               # target nuclus (Not used: needed only for my own module to plot)
L16:inpname="sample_ncsmdata.dat"    # input file
L17:qT=1.e-1                # scale factor of proposal for hyperparameters
L18:qY=1.e-3                # scale factor of proposal for predictions y*
L19:qYfac = 5.e-3           # reduction factor of qY after "Resampling"
L20:xpMax=46                # Maximum Nmax to be calculated. Start with large number and then decrease if you don't need the large number.

<sample_ncsmdata.dat>
The code now only supports the following format of FCI results.

[Nmax],[hbar omega],[g.s. Energy]


### How to run
1. Modify "input.jl"
2. $julia bayesCGP_v1.jl

### Outputs
1. Posterior*dat:       #Posterior distributions for y* (mean and stds are summarized)
2. Thetas*.dat:         #Hyperparameter distributions after mstep iteration of Metropolis-Hastings updates
