export SparseELL1, SparseELL2

"""
    SparseELL1{Tv,Ti}

Type that represents an ELL (Ellpack) format of sparse matrix from CSR format.

### Fileds
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `k::Int`: the maximum number of non-zero elements in row
- `idx::Matrix{Ti}`: a matrix of column indices. 0 indicates there is no corresponding elements.
- `val::Matrix{Tv}`: a matrix of non-zero elements
"""
struct SparseELL1{Tv,Ti}
    m::Ti
    n::Ti
    k::Int
    idx::Array{Ti,2}
    val::Array{Tv,2}
end

"""
    SparseELL2{Tv,Ti}

Type that represents an ELL (Ellpack) format of sparse matrix from CSC format.

### Fileds
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `k::Int`: the maximum number of non-zero elements in column
- `idx::Matrix{Ti}`: a matrix of row indices. 0 indicates there is no corresponding elements.
- `val::Matrix{Tv}`: a matrix of non-zero elements
"""
struct SparseELL2{Tv,Ti}
    m::Ti
    n::Ti
    k::Int
    idx::Array{Ti,2}
    val::Array{Tv,2}
end

"""
    SparseELL1(A::Matrix{Tv}) where Tv
    SparseELL1(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseELL1(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseELL1(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseELL1(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Create a sparse matrix with ELL format (row-based).
"""
function SparseELL1(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m = A.m
    n = A.n
    k = maximum(diff(A.rowptr))
    elem = zeros(Tv, m, k)
    idx = zeros(Ti, m, k)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            v = z - A.rowptr[i] + 1
            j = A.colind[z]
            idx[i,v] = j
            elem[i,v] = A.val[z]
        end
    end
    SparseELL1{Tv,Ti}(m, n, k, idx, elem)
end

"""
    SparseELL2(A::Matrix{Tv}) where Tv
    SparseELL2(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseELL2(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseELL2(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseELL2(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

Create a sparse matrix with ELL format (column-based).
"""
function SparseELL2(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m = A.m
    n = A.n
    k = maximum(diff(A.colptr))
    elem = zeros(Tv, n, k)
    idx = zeros(Ti, n, k)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            v = z - A.colptr[j] + 1
            i = A.rowind[z]
            idx[j,v] = i
            elem[j,v] = A.val[z]
        end
    end
    SparseELL2{Tv,Ti}(m, n, k, idx, elem)
end

SparseELL1(A::Matrix{Tv}) where Tv = SparseELL1(SparseCSR(A))
SparseELL1(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = SparseELL1(SparseCSR(A))
SparseELL1(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = SparseELL1(SparseCSR(A))
SparseELL1(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = SparseELL1(SparseCSR(A))

SparseELL2(A::Matrix{Tv}) where Tv = SparseELL2(SparseCSC(A))
SparseELL2(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = SparseELL2(SparseCSC(A))
SparseELL2(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = SparseELL2(SparseCSC(A))
SparseELL2(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = SparseELL2(SparseCSC(A))
