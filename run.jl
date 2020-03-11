using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
using Base.Threads
using LsqFit
using SIMD
using Glob 
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
updatefunc = y_corr
#updatefunc = y_Gibbs ## too slow
#inps = glob("./data/*.txt")
inps = glob("sample_ncsmdata.dat")

for paramean in [T,F]
    if paramean==T
        qT = 1.e-1; qY = 1.e-2; qYfac = 1.e-2
    else
        qT = 1.e-1; qY = 1.e-3; qYfac = 1.e-2
    end
    for inpname in inps
        if occursin("N3LO",inpname); inttype="N3LO"
        elseif occursin("JISP16",inpname); inttype="JISP16"
        elseif occursin("NNLOopt",inpname); inttype="NNLOopt"
        else; inttype="unknownint";end
        println("inttype $inttype parametric mean:$paramean ")
        print("  numN:$numN step:$mstep Kernel:$Kernel ")
        print("Monotonic:$Monotonic Convex:$Convex ")
        if inttype=="N3LO"
            xpMax = 38
        else
            xpMax = 46
        end
        @time Em,Ev=main(
            mstep,numN,sigRfac,ktype,
            updatefunc,inpname,inttype,
            xpMax,Monotonic,Convex,paramean,
            qT,qY,qYfac)
    end
end

