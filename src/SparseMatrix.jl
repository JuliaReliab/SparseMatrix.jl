module SparseMatrix

import LinearAlgebra #: Adjoint
import SparseArrays #: SparseMatrixCSC, nnz

export AbstractSparseM
export SparseCSR, SparseCSC, SparseCOO, BlockCOO
export spdiag
export block
export fill!, scal!, axpy!, gemv!, spger!, gemm!

include("_sparse.jl")
include("_blas_level1.jl")
include("_blas_level2.jl")
include("_blas_level3.jl")

include("_mul.jl")
include("_spdiag.jl")
include("_block.jl")


end
