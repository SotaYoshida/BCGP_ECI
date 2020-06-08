# MvNormal!.jl

# The MIT License (MIT)
# Copyright (c) 2020 Sota Yoshida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


## A "destructive" implementation of sampling from a multivariate normal to reduce memory usage.

## mean: mean vector
## lt: length of mean
## cqY: arbitrary factor multiplied to the covariance
## zerovec: pre-allocated vector to store the sample
## cLL: cholesky factor of the covariance matrix
## rv: [randn() for i=1:lt]

function MvNormal!(mean::Array{Float64,1},lt::Int64,cqY::Float64,
                   zerovec::Array{Float64,1},
                   cLL::LowerTriangular{Float64,Array{Float64,2}},
                   rv::Array{Float64,1})
    for i= 1:lt
        zerovec[i] = mean[i]
        for j=1:i
            zerovec[i] += cqY* cLL[i,j] * rv[j]
        end
    end
    nothing
end
