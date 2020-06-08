# Cholesky!.jl

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


## A "destructive" implementation of cholesky decomposition to reduce memory usage.
## Sigma: PSD matrix
## ln: Dim. of the matrix
## cLL: pre-allocated lower triangular matrix to store cholesly factor 

function Cholesky!(Sigma::Array{Float64,2},
                     ln::Int64,cLL::LowerTriangular{Float64,Array{Float64,2}})
    l11 = sqrt(Sigma[1,1]) 
    cLL[1,1] = l11
    cLL[2,1] = Sigma[2,1]/l11; cLL[2,2] = sqrt( Sigma[2,2]-cLL[2,1]^2)
    for i=3:ln
        for j=1:i-1
            cLL[i,j] = Sigma[i,j]
            @simd for k = 1:j-1
                cLL[i,j] += - cLL[i,k]*cLL[j,k]                
            end
            cLL[i,j] = cLL[i,j] / cLL[j,j]            
        end
        cLL[i,i] = Sigma[i,i]
        @simd for j=1:i-1
            cLL[i,i] += -cLL[i,j]^2
        end
        cLL[i,i] = sqrt(cLL[i,i])             
    end
    nothing
end
