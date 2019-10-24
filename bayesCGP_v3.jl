using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
using Base.Threads
using LsqFit
using SIMD
#using BenchmarkTools
include("input.jl")
include("BayesGPsubmodule_v3.jl")

println("input file:$inpname")
println("numN:$numN step:$mstep\n\n")
println("parametric mean is used:",paramean)

if occursin("N3LO",inpname); inttype="N3LO"
elseif occursin("JISP16",inpname); inttype="JISP16"
elseif occursin("NNLOopt",inpname); inttype="NNLOopt"
else; inttype="unknownint";end
println("inttype $inttype")

@time Em,Ev=main(mstep,numN,sigRfac)
println("   |-> time for main()")

