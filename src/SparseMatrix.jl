module SparseMatrix

import Base #: size, length, show, +, -, *, /, adjoint, convert, getindex, iterate
using LinearAlgebra
using SparseArrays

include("_SparseMatrix.jl")

end
