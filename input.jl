numN=2000 ## # of particle > 20,000 is recommended!
mstep=2000 ## # of iteration of MC sampling
Rstep=500 ## # of iteration of MC sampling to determine R_E
Kernel="logMatern" ##### "Lin+Mat"/"RQ"/"RBF"/"Matern"/"Mat32"/"logRBF"/"logMat32"/"NSMatern"/"logRQ"
sigfac=1 ##  R_E = Rmean + sigfac * Rstd
Auxiliary=false
Monotonic=false
Convex=false
fixed=false
printder=false
multihw=false
rMH=[0.30,0.30]
CImode ="NCSM" ### "NCSM", "MCSM"

nuc="6Li"
inpname="sample_ncsmdata.dat"
qT=1.e-1;qY=1.e-3;qYfac = 5.e-3;xpMax=46
