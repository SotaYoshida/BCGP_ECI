numN=2000 ## # of particle 50,000 is used in the present work
mstep=3000 ## # of iteration of MC sampling
Rstep=500 ## # of iteration of MC sampling to determine R_E
Kernel="logMatern" ##### "Lin+Mat"/"RQ"/"RBF"/"Matern"/"Mat32"/"logRBF"/"logMat32"/"NSMatern"/"logRQ"
sigfac=1 ##  R_E = Rmean + sigfac * Rstd
Auxiliary=false
Monotonic=false
Convex=false
fixed=false
printder=false
multihw=false
qT=0.3;qY=1.e-2 ### for sparse data
#qT=1.e-1;qY=1.e-2;qYfac=1.e-4 ### for N3LO 6Li
qT=1.e-1;qY=1.e-2;qYfac = 1.e-3 ### for JISP/NNLOopt
rMH=[0.30,0.30]


inpname="sample_ncsmdata.dat"
nuc="6Li"
CImode ="NCSM" ### "NCSM", "MCSM"