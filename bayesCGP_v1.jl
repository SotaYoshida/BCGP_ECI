using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
include("./input.jl")
include("./submodule/BayesGPsubmodule.jl")


ENV["PYTHON"]="python3.7"
ENV["PLOTS_DEFAULT_BACKEND"] = "PyPlot"
#using PyCall
#using PyPlot
#include("./ownplot.jl")
#using .ownplot

println("input file:$inpname")
## Initialize global variables
Tsigma,tTheta,xtrain,ytrain,xprd,xun,yun,oxtrain,oytrain,iThetas,lt,lp,Mysigma,muy,mstd=readinput(inpname)

R,sigR=detR(xtrain,ytrain,yun,mstep,Mysigma,muy,mstd)

Em,Ev=main(xtrain,ytrain,yun,mstep,numN,Mysigma,muy,mstd)



