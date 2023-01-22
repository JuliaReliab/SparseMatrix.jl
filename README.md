# SparseMatrix

[![Build Status](https://travis-ci.com/okamumu/SparseMatrix.jl.svg?branch=master)](https://travis-ci.com/okamumu/SparseMatrix.jl)
[![Codecov](https://codecov.io/gh/okamumu/SparseMatrix.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/okamumu/SparseMatrix.jl)
[![Coveralls](https://coveralls.io/repos/github/okamumu/SparseMatrix.jl/badge.svg?branch=master)](https://coveralls.io/github/okamumu/SparseMatrix.jl?branch=master)

SparseMatrix.jl provides the sparse matrix with CSR, CSC and COO formats, and defines BLAS-like routines for their basic linar
algebra computations. In addition, it provides a simple block matrix.

## Installation

This is not in the official Julia package yet. Please run the following command to install it.
```
using Pkg; Pkg.add(PackageSpec(url="https://github.com/JuliaReliab/SparseMatrix.jl.git"))
```
This package depends on the packages; `LinearAlgebra`, `SparseArrays`

## Load module

Load the module:
```
using SparseMatrix
```

## Create SparseMatrix

The package provides the sparse matrix with CSR, CSC and COO formats. These can be coverted from
Matrix and SparseArrays.SparseMatrixCSC. In addition, they are converted from each others.

```julia
using SparseMatrix
using SparseArrays

A = [1.0 0.0; 0.0 4.0]
csrA = SparseCSR(A)
cscA = SparseCSC(A)
cooA = SparseCOO(A)

B = spzeros(2,2)
B[1,1] = 1.0
B[2,2] = 4.0
csrB = SparseCSR(B)
cscB = SparseCSC(B)
cooB = SparseCOO(B)
```

## BLAS

The package provides the following blas routines for SparseCSR, SparseCSC and SparseCOO where A is one of these matrices.

- Level 1
```julia
fill!(v, A) ## zero elements are ignored (the structure of zero elements are not changed)
scal!(alpha, A)
axpy!(alpha, X, Y)
```
- Level 2
```julia
gemv!(trA, alpha, A, x, beta, y) ## y = alpha * A * x + beta * y where trA is 'N' or 'T'
spger!(alpha, x, y, beta, A) ## A = alpha * x * y + beta * A where zero elements are ignored.
```
- Level 3
```julia
gemm!(trA, trB, alpha, A, X, beta, Y) ## Y = alpha * A * B + beta * C where tr is 'N' or 'T'
```

Note that `spger!` is similar to `ger` in the usual BLAS but it ignores the zero elements.
In `gemm!`, either A or B is SparseCSR, SparseCSC and SparseCOO.

## Block Matrix

`BlockCOO` is a block matrix based on COO formant. This can be extracted to SparseCSR, SparseCSC and
SparseCOO whose elements are Float64. BLAS for BlockCOO has not been implemented yet. The function `block` is to
create `BlockCOO` from Matrix and SparseArrays.SparseMatrixCSC whose elements are Matrix.

```julia
using SparseMatrix
using SparseArrays

A = [-2 2; 3 -3]
B = [-10 10; 1 -1]
M = BlockCOO(2, 2, [(1,1,A), (2,2,B)])
csrM = SparseCSR(M)
cscM = SparseCSC(M)
cooM = SparseCOO(M)

N = spzeros(AbstractMatrix{Int}, 2,2)
N[1,1] = A
N[2,2] = B
csrN = SparseCSR(block(N))
cscN = SparseCSC(block(N))
cooN = SparseCOO(block(N))
```
