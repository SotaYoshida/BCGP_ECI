using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
using Base.Threads
using LsqFit
using BenchmarkTools
include("./input.jl")
include("./submodule/BayesGPsubmodule_v2.jl")
#include("./submodule/KernelSelection.jl")
#OMP_NUM_THREADS=4
#ENV["PYTHON"]="python3.7"
#ENV["PLOTS_DEFAULT_BACKEND"] = "PyPlot"

#using PyCall
#using PyPlot

println("input file:$inpname")
println("numN:$numN step:$mstep\n\n")
println("parametric mean is used:",paramean)

if occursin("N3LO",inpname); inttype="N3LO"
elseif occursin("JISP16",inpname); inttype="JISP16"
elseif occursin("NNLOopt",inpname); inttype="NNLOopt"
else; inttype="unknownint";end
println("inttype $inttype")
## Initialize global variables
Tsigma,tTheta,xtrain,ytrain,xprd,xun,yun,oxtrain,oytrain,iThetas,lt,lp,Mysigma,muy,mstd,pfit,Rtt,Rpt,Rpp=readinput(inpname)
tKtp=zeros(Float64,lt,lp);tKpt=zeros(Float64,lp,lt)

# R=0.0;sigR=0.0
# Kernels=["logMatern","logMat32"]
# qT=5.0;qY=5.0;qYfac = 1.0
# KernelSelection(Kernels,xtrain,ytrain,yun,mstep,Mysigma,muy,mstd)
# exit()

# R=0.0;sigR=0.0
# @time R,sigR=detR(xtrain,ytrain,yun,mstep,Mysigma,muy,mstd)
# println("   |-> time for detR()")
# #exit()

R=(ytrain[lt-1]-ytrain[lt])/(ytrain[lt-2]-ytrain[lt-1])
sigR = 0.1 * R
@time Em,Ev=main(xtrain,ytrain,yun,mstep,numN,Mysigma,muy,mstd)
println("   |-> time for main()")

