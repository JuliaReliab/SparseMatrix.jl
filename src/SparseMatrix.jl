module SparseMatrix

using LinearAlgebra #: Adjoint
import SparseArrays #: SparseMatrixCSC, nnz

export AbstractSparseM
export SparseCSR, SparseCSC, SparseCOO, BlockCOO
export spdiag

include("_sparse.jl")
include("_mul.jl")
include("_spdiag.jl")
include("_block.jl")

end
