export AbstractSparseM
export SparseCSR, SparseCSC, SparseCOO
export nnz

import Base: iszero

function iszero(x::Float64)
    x ≈ 0.0
end

abstract type AbstractSparseM{Tv, Ti} <: AbstractMatrix{Tv} end

struct SparseCSR{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    rowptr::Vector{Ti}
    colind::Vector{Ti}
end

struct SparseCSC{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    colptr::Vector{Ti}
    rowind::Vector{Ti}
end

struct SparseCOO{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    rowind::Vector{Ti}
    colind::Vector{Ti}
end

SparseCSR(A::Matrix{Tv}) where {Tv} = _tocsr(A, Int)
SparseCSR(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCSR(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(A))
SparseCSR(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsr(A)
SparseCSR(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(_tocsc(A)))

SparseCSC(A::Matrix{Tv}) where {Tv} = _tocsc(A, Int)
SparseCSC(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocsc(_tocoo(A))
SparseCSC(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCSC(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsc(A)
SparseCSC(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsc(A)

SparseCOO(A::Matrix{Tv}) where {Tv} = _tocoo(A, Int)
SparseCOO(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCOO(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(_tocsc(A))

Matrix(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _dense(A)
Matrix(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _dense(A)
Matrix(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _dense(A)

function SparseCOO(m::Ti, n::Ti, elem::AbstractArray{Tuple{Ti,Ti,Tv},1}) where {Tv,Ti}
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    prev_index::Tuple{Ti,Ti} = (0,0)
    for (i,j,u) in sort(elem)
        if m < i
            m = i
        end
        if n < j
            n = j
        end
        if prev_index != (i,j)
            push!(rowind, i)
            push!(colind, j)
            push!(val, u)
            prev_index = (i,j)
        else
            val[end] += u
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

### overload

function Base.show(io::IO, A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(m)*"x"*string(n)*" CSR-SparseMatrix")
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" CSC-SparseMatrix")
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" COO-SparseMatrix")
    for z = 1:nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(m)*"x"*string(n)*" CSR-SparseMatrix")
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" CSC-SparseMatrix")
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" COO-SparseMatrix")
    for z = 1:nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
    end
end

function nnz(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return length(A.val)
end

####

function Base.size(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return (A.m, A.n)
end

function Base.eachindex(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return eachindex(A.val)
end

function Base.length(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return length(A.val)
end

function Base.getindex(A::AbstractSparseM{Tv,Ti}, i::Ti) where {Tv,Ti}
    A.val[i]
end

function Base.setindex!(A::AbstractSparseM{Tv,Ti}, value::Tv, i::Ti) where {Tv,Ti}
    A.val[i] = value
end

# function Base.iterate(A::AbstractSparseM{Tv,Ti}, i::Ti = 1) where {Tv,Ti}
#     i == length(A)+1 && return nothing
#     return (A.val[i], i+1)
# end

####

function Base.copy(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, copy(A.val), A.rowptr, A.colind)
end

function Base.copy(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, copy(A.val), A.colptr, A.rowind)
end

function Base.copy(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, copy(A.val), A.rowind, A.colind)
end

####

function Base.similar(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, similar(A.val), A.rowptr, A.colind)
end

function Base.similar(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, similar(A.val), A.colptr, A.rowind)
end

function Base.similar(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, similar(A.val), A.rowind, A.colind)
end

####

function Base.zero(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, zero(A.val), A.rowptr, A.colind)
end

function Base.zero(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, zero(A.val), A.colptr, A.rowind)
end

function Base.zero(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, zero(A.val), A.rowind, A.colind)
end

function _tocsr(A::Matrix{Tv}, ::Type{Ti})::SparseCSR{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowptr = Vector{Ti}(undef, m+1)
    colind = Vector{Ti}()
    val = Vector{Tv}()
    rowptr[1] = 1
    for i = 1:m
        for j = 1:n
            if !(iszero(A[i,j]))
                push!(colind, j)
                push!(val, A[i,j])
            end
        end
        rowptr[i+1] = length(val)+1
    end
    SparseCSR(m, n, val, rowptr, colind)
end

function _tocsr(A::SparseCOO{Tv,Ti})::SparseCSR{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowptr = Vector{Ti}(undef, m+1)
    colind = Vector{Ti}()
    val = Vector{Tv}()
    p = 1
    i = 1
    rowptr[i] = p
    for x in sort(collect(zip(A.rowind, A.colind, A.val)))
        if i != x[1]
            for u = i+1:x[1]
                rowptr[u] = p
            end
            i = x[1]
        end
        push!(colind, x[2])
        push!(val, x[3])
        p += 1
    end
    for u = i+1:m+1
        rowptr[u] = p
    end
    SparseCSR(m, n, val, rowptr, colind)    
end

function _tocsc(A::Matrix{Tv}, ::Type{Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    colptr = Vector{Ti}(undef, n+1)
    rowind = Vector{Ti}()
    val = Vector{Tv}()
    colptr[1] = 1
    for j = 1:n
        for i = 1:m
            if !(iszero(A[i,j]))
                push!(rowind, i)
                push!(val, A[i,j])
            end
        end
        colptr[j+1] = length(val)+1
    end
    SparseCSC(m, n, val, colptr, rowind)
end

function _tocsc(A::SparseCOO{Tv,Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    colptr = Vector{Ti}(undef, n+1)
    rowind = Vector{Ti}()
    val = Vector{Tv}()
    p = 1
    j = 1
    colptr[j] = p
    for x in sort(collect(zip(A.colind, A.rowind, A.val)))
        if j != x[1]
            for u = j+1:x[1]
                colptr[u] = p
            end
            j = x[1]
        end
        push!(rowind, x[2])
        push!(val, x[3])
        p += 1
    end
    for u = j+1:n+1
        colptr[u] = p
    end
    SparseCSC(m, n, val, colptr, rowind)
end

function _tocsc(A::SparseMatrixCSC{Tv,Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
    SparseCSC(A.m, A.n, copy(A.nzval), copy(A.colptr), copy(A.rowval))
end

function _tocoo(A::Matrix{Tv}, ::Type{Ti})::SparseCOO{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    for j = 1:n
        for i = 1:m
            if !(iszero(A[i,j]))
                push!(rowind, i)
                push!(colind, j)
                push!(val, A[i,j])
            end
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

function _tocoo(A::SparseCSR{Tv,Ti})::SparseCOO{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            push!(rowind, i)
            push!(colind, j)
            push!(val, A.val[z])
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

function _tocoo(A::SparseCSC{Tv,Ti})::SparseCOO{Tv,Ti} where {Tv, Ti}
    m, n = size(A)
    rowind = Vector{Ti}()
    colind = Vector{Ti}()
    val = Vector{Tv}()
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            push!(rowind, i)
            push!(colind, j)
            push!(val, A.val[z])
        end
    end
    SparseCOO(m, n, val, rowind, colind)
end

function _dense(A::SparseCSR{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(m,n)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            M[i,j] = A.val[z]
        end
    end
    return M
end

function _dense(A::SparseCSC{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(m,n)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            M[i,j] = A.val[z]
        end
    end
    return M
end

function _dense(A::SparseCOO{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(m,n)
    for z = 1:nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        M[i,j] = A.val[z]
    end
    return M
end

#################### Adjoint

struct Adjoint{T <: AbstractSparseM}
    parent::T
end

function Base.adjoint(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    Adjoint(A)
end

function Base.adjoint(A::Adjoint{T}) where {T}
    A.parent
end

