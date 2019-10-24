numN=20000 ## # of particle 20,000 is used in the present work
mstep=3000 ## # of iteration of MC sampling
sigRfac = 0.1

numN=5000
mstep=1000

Kernel="logMatern"

Kernelder=false 
Monotonic=true;Convex=true
#Monotonic=true;Convex=false ## wo_C
#Monotonic=false;Convex=false ## wo_MC

#paramean=true    ### parametric mean:B3 fit
paramean=false  

fixed=false;printder=false;multihw=false
rMH=[0.30,0.30]
CImode ="NCSM" ## "MCSM" 

nuc="6Li"

inpname="sample_ncsmdata.dat"

if paramean==false
    qT=1.e-1;qY=1.e-3;qYfac = 1.e-3;xpMax=40 ### 
    #qT=1.e-1;qY=1.e-3;qYfac = 1.e-3;xpMax=46 ### for JISP/NNLOopt
else
    qT=8.e-2;qY=1.e-3;qYfac=1.0;xpMax=46
    qT=6.e-2;qY=1.e-3;qYfac=0.5;xpMax=46
end
if Convex==false; qY=0.5;qYfac=1.0;end

