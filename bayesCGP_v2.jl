using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
using Base.Threads
include("./input.jl")
include("./submodule/BayesGPsubmodule_v2.jl")


ENV["PYTHON"]="python3.7"
ENV["PLOTS_DEFAULT_BACKEND"] = "PyPlot"
#using PyCall
#using PyPlot
#include("./ownplot.jl")
#using .ownplot

println("input file:$inpname")
println("numN:$numN step:$mstep\n\n")

if occursin("N3LO",inpname); inttype="N3LO"
elseif occursin("JISP16",inpname); inttype="JISP16"
elseif occursin("NNLOopt",inpname); inttype="NNLOopt"
else; inttype="unknownint";end
println("inttype $inttype")
## Initialize global variables
Tsigma,tTheta,xtrain,ytrain,xprd,xun,yun,oxtrain,oytrain,iThetas,lt,lp,Mysigma,muy,mstd=readinput(inpname)

R=0.0;sigR=0.0
@time R,sigR=detR(xtrain,ytrain,yun,mstep,Mysigma,muy,mstd)
println("   |-> time for detR()")

#const R=0.5137080436573149; const sigR=0.1100703232765452
@time Em,Ev=main(xtrain,ytrain,yun,mstep,numN,Mysigma,muy,mstd)
println("   |-> time for main()")


