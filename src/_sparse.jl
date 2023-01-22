import LinearAlgebra
import SparseArrays

export AbstractSparseM
export SparseCSR, SparseCSC, SparseCOO, BlockCOO

function Base.iszero(x::Float64)
    x ≈ 0.0
end

"""
    AbstractSparseM{Tv,Ti} <: AbstractMatrix{Tv}

Abstract tyoe for sparse matrix.

### Notes

Every concrete `AbstractSparseM` must have the following fields:
- `m`: the number of rows whose type is Ti
- `n`: the number of columns whose type is Ti
- `val`: a vector of non-zero elements whose type is Tv
"""
abstract type AbstractSparseM{Tv,Ti} <: AbstractMatrix{Tv} end

"""
    SparseCSR{Tv,Ti} <: AbstractSparseM{Tv,Ti}

Type that represents a sparse matrix with CSR format.

### Fileds
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `val::Vector{Tv}`: a vector of non-zero elements whose type is Tv
- `rowptr::Vector{Ti}`: a vector to indicate a position of `val` to start each row.
- `colind::Vector{Ti}`: a vector indicating the column index for the corredponding element of `val`.
"""
struct SparseCSR{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    rowptr::Vector{Ti}
    colind::Vector{Ti}
end

"""
    SparseCSC{Tv,Ti} <: AbstractSparseM{Tv,Ti}

Type that represents a sparse matrix with CSC format.

### Fileds
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `val::Vector{Tv}`: a vector of non-zero elements whose type is Tv
- `colptr::Vector{Ti}`: a vector to indicate a position of `val` to start each column.
- `rowind::Vector{Ti}`: a vector indicating the row index for the corredponding element of `val`.
"""
struct SparseCSC{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    colptr::Vector{Ti}
    rowind::Vector{Ti}
end

"""
    SparseCOO{Tv,Ti} <: AbstractSparseM{Tv,Ti}

Type that represents a sparse matrix with COO format.

### Fileds
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `val::Vector{Tv}`: a vector of non-zero elements whose type is Tv
- `rowind::Vector{Ti}`: a vector indicating the row index for the corredponding element of `val`.
- `colind::Vector{Ti}`: a vector indicating the column index for the corredponding element of `val`.
"""
struct SparseCOO{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{Tv}
    rowind::Vector{Ti}
    colind::Vector{Ti}
end

"""
    SparseCSR(A)

Create a sparse matrix with CSR format from the matrix `A`.
The matrix `A` is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
SparseCSR(A::Matrix{Tv}) where {Tv} = _tocsr(A, Int)
SparseCSR(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCSR(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(A))
SparseCSR(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsr(A)
SparseCSR(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(_tocsc(A)))

"""
    SparseCSC(A)

Create a sparse matrix with CSC format from the matrix `A`.
The matrix `A` is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
SparseCSC(A::Matrix{Tv}) where {Tv} = _tocsc(A, Int)
SparseCSC(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocsc(_tocoo(A))
SparseCSC(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCSC(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsc(A)
SparseCSC(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsc(A)

"""
    SparseCOO(A)

Create a sparse matrix with COO format from the matrix `A`.
The matrix `A` is allowed to be Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
SparseCOO(A::Matrix{Tv}) where {Tv} = _tocoo(A, Int)
SparseCOO(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = copy(A)
SparseCOO(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(_tocsc(A))

"""
    Matrix(A)

Create a dense matrix Matrix{Tv} from the matrix `A`.
The matrix `A` is allowed to be SparseCSR, SparseCSC, SparseCOO and BlockCOO.
"""
Matrix(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _todense(A)
Matrix(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _todense(A)
Matrix(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _todense(A)

SparseArrays.sparse(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = SparseArrays.sparse(_tocsc(_tocoo(A)))
SparseArrays.sparse(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = SparseArrays.SparseMatrixCSC{Tv,Ti}(A.m, A.n, copy(A.colptr), copy(A.rowind), copy(A.val))
SparseArrays.sparse(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = SparseArrays.sparse(_tocsc(A))

"""
    SparseCOO(m::Ti, n::Ti, elem::AbstractArray{Tuple{Ti,Ti,Tv},1}) where {Tv,Ti}

Create a m-by-n sparse matrix with COO format from the list of tuple (rowindex, colindex, value).
"""
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
    for z = 1:SparseArrays.nnz(A)
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
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
    end
end

function SparseArrays.nnz(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
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

function Base.zero(::Type{AbstractMatrix{Tv}}) where Tv
    0
end

function Base.zero(A::SparseCSR{Tv,Ti}) where {Tv,Ti}
    SparseCSR(A.m, A.n, zero(A.val), A.rowptr, A.colind)
end

function Base.zero(A::SparseCSC{Tv,Ti}) where {Tv,Ti}
    SparseCSC(A.m, A.n, zero(A.val), A.colptr, A.rowind)
end

function Base.zero(A::SparseCOO{Tv,Ti}) where {Tv,Ti}
    SparseCOO(A.m, A.n, zero(A.val), A.rowind, A.colind)
end

function Base.zero(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    SparseArrays.SparseMatrixCSC{Tv,Ti}(A.m, A.n, A.colptr, A.rowval, zero(A.nzval))
end

###

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

function _tocsc(A::SparseArrays.SparseMatrixCSC{Tv,Ti})::SparseCSC{Tv,Ti} where {Tv, Ti}
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

function _todense(A::SparseCSR{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
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

function _todense(A::SparseCSC{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
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

function _todense(A::SparseCOO{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, n = size(A)
    M = zeros(m,n)
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        M[i,j] = A.val[z]
    end
    return M
end
