using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
using Base.Threads
using LsqFit
using SIMD
using Glob 
#include("shiftcheck_cgp_main.jl")
include("cgp_main.jl")
T=true;F=false

ktype = logMat52  ## Mat52/logMat52/Mat32/logMat32/RBF/logRBF
Kernel= string(ktype)
multihw=false
const sigRfac = 0.1
const numN = 20000
const mstep = 3000

Monotonic=T
Convex= T

#inps = glob("./data/*.txt")
inps = glob("sample_ncsmdata.dat")
#inps = glob("./data/ncsm_6Li_hw16_lam202_N3LO_6-14.txt")
#inps = glob("./data/Shin_6Li_ncsm_JISP16_2-18.txt")

for paramean in [F,T]
    if paramean==T
        qT = 2.e-1; qY = 1.e-2; qYfac = 5.e-2
    else
        qT = 2.e-1; qY = 1.e-2; qYfac = 1.e-1
    end
    for inpname in inps
        if occursin("N3LO",inpname); inttype="N3LO"
        elseif occursin("JISP16",inpname); inttype="JISP16"
        elseif occursin("NNLOopt",inpname); inttype="NNLOopt"
        else; inttype="unknownint";end
        println("inttype $inttype parametric mean:$paramean ")
        print(" numN:$numN step:$mstep Kernel:$Kernel")
        print(" Monotonic:$Monotonic Convex:$Convex ")
        if inttype=="N3LO"
            xpMax = 38
        else
            xpMax = 46
        end
        @time Em,Ev=main(
            mstep,numN,sigRfac,ktype,
            inpname,inttype,
            xpMax,Monotonic,Convex,paramean,
            qT,qY,qYfac)
    end
end

