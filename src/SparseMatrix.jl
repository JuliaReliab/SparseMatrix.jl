module SparseMatrix

import Base #: size, length, show, +, -, *, /, adjoint, convert, getindex, iterate
using LinearAlgebra
using SparseArrays: SparseMatrixCSC

include("_sparse.jl")
include("_mul.jl")
include("_spdiag.jl")

include("_block.jl")

end
