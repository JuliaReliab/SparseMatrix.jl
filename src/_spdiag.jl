export spdiag

#################### Diag

struct Diag{Tv <: AbstractFloat, Ti <: Integer} <: AbstractArray{Tv,1}
    index::Vector{Ti}
    val::Vector{Tv}
end

function spdiag(A::Matrix{Tv}) where {Tv}
    m, n = size(A)
    index = Vector{Int}(undef, min(m,n))
    z = 1
    for i = 1:min(m,n)
        index[i] = z
        z += m+1
    end
    val = reshape(A, length(A))
    Diag(index, val)
end

function spdiag(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    index = zeros(Ti, min(m,n))
    val = reshape(A.val, length(A.val))
    for i = 1:min(m,n)
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
    for j = 1:min(m,n)
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
    for z = 1:nnz(A)
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

function Base.getindex(A::Diag{Tv,Ti}, i::Ti) where {Tv,Ti}
    z = A.index[i]
    if z == 0
        return Tv(0)
    else
        return A.val[z]
    end
end

function Base.setindex!(A::Diag{Tv,Ti}, value::Tv, i::Ti) where {Tv,Ti}
    z = A.index[i]
    if z != 0
        A.val[z] = value
    end
end
