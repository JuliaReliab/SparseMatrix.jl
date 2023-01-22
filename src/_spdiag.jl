export spdiag

struct Diag{Tv <: AbstractFloat, Ti <: Integer} <: AbstractArray{Tv,1}
    index::Vector{Ti}
    val::Vector{Tv}
end

"""
    spdiag(A)

Create a type of Diag to represent the diagonal matrix of A.
The type of A is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR,
SparseCSC, SparseCOO.

### Input

- `A` -- an instance of matrix; Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO.

### Output

An instance of Diag which a structure.
"""
function spdiag(A::Matrix{Tv}) where {Tv}
    m, n = size(A)
    index = Vector{Int}(undef, min(m,n))
    z = 1
    @inbounds for i = 1:min(m,n)
        index[i] = z
        z += m+1
    end
    val = reshape(A, length(A))
    Diag(index, val)
end

function spdiag(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.nzval, length(A.nzval))
    @inbounds for j = 1:min(m,n)
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[z]
            if i == j
                index[i] = z
                break
            elseif i > j
                break
            end
        end
    end
    Diag(index, val)
end

function spdiag(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    @inbounds for i = 1:min(m,n)
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            if i == j
                index[i] = z
                break
            elseif i < j
                break
            end
        end
    end
    Diag(index, val)
end

function spdiag(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    @inbounds for j = 1:min(m,n)
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            if i == j
                index[i] = z
                break
            elseif i > j
                break
            end
        end
    end
    Diag(index, val)
end

function spdiag(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    @inbounds for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        if i == j
            index[i] = z
        end
    end
    Diag(index, val)
end

function Base.length(A::Diag{Tv,Ti}) where {Tv,Ti}
    return length(A.index)
end

function Base.size(A::Diag{Tv,Ti}) where {Tv,Ti}
    return (length(A.index),)
end

@inbounds function Base.getindex(A::Diag{Tv,Ti}, i::Ti) where {Tv,Ti}
    z = A.index[i]
    if z == 0
        return Tv(0)
    else
        return A.val[z]
    end
end

@inbounds function Base.setindex!(A::Diag{Tv,Ti}, value::Tv, i::Ti) where {Tv,Ti}
    z = A.index[i]
    if z != 0
        A.val[z] = value
    else
        @warn "Warning: There does not exist the index $i in the sparse matrix. " *
                "Probably the diagonal element was 0."
    end
end
