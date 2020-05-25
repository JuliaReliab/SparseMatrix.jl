import Base #: size, length, show, +, -, *, /, adjoint, convert, getindex, iterate
using LinearAlgebra
using SparseArrays

export AbstractSparseM
export SparseCSR, SparseCSC, SparseCOO
export nnz, spdiag
export sprand, sprandn

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
SparseCSR(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(A))
SparseCSR(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsr(A)
SparseCSR(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(_tocsc(A)))
SparseCSC(A::Matrix{Tv}) where {Tv} = _tocsc(A, Int)
SparseCSC(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocsc(_tocoo(A))
SparseCSC(A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _tocsc(A)
SparseCSC(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocsc(A)
SparseCOO(A::Matrix{Tv}) where {Tv} = _tocoo(A, Int)
SparseCOO(A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseCOO(A::SparseArrays.SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = _tocoo(_tocsc(A))
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

function Base.size(A::AbstractSparseM{Tv,Ti}) where {Tv,Ti}
    return (A.m, A.n)
end

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
            if !(A[i,j] ≈ 0.0)
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
            if !(A[i,j] ≈ 0.0)
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
            if !(A[i,j] ≈ 0.0)
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

#################### CSR

function _mul(A::SparseCSR{Tv,Ti}, x::Vector{Tv})::Vector{Tv} where {Tv, Ti}
    m, n = size(A)
    y = zeros(m)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            j = A.colind[z]
            y[i] += A.val[z] * x[j]
        end
    end
    return y
end

function _mul(At::Adjoint{SparseCSR{Tv,Ti}}, x::Vector{Tv})::Vector{Tv} where {Tv, Ti}
    A = At.parent
    n, m = size(A)
    y = zeros(m)
    for j = 1:n
        for z = A.rowptr[j]:A.rowptr[j+1]-1
            i = A.colind[z]
            y[i] += A.val[z] * x[j]
        end
    end
    return y
end

###

function _mul(A::SparseCSR{Tv,Ti}, B::Matrix{Tv})::Matrix{Tv} where {Tv, Ti}
    m, k = size(A)
    k, n = size(B)
    C = zeros(m,n)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            l = A.colind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[l,j]
            end
        end
    end
    return C
end

function _mul(A::SparseCSR{Tv,Ti}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    m, k = size(A)
    n, k = size(B)
    C = zeros(m,n)
    for i = 1:m
        for z = A.rowptr[i]:A.rowptr[i+1]-1
            l = A.colind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[j,l]
            end
        end
    end
    return C
end

function _mul(At::Adjoint{SparseCSR{Tv,Ti}}, B::Matrix{Tv})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    k, m = size(A)
    k, n = size(B)
    C = zeros(m,n)
    for l = 1:k
        for z = A.rowptr[l]:A.rowptr[l+1]-1
            i = A.colind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[l,j]
            end
        end
    end
    return C
end

function _mul(At::Adjoint{SparseCSR{Tv,Ti}}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    B = Bt.parent
    k, m = size(A)
    n, k = size(B)
    C = zeros(m,n)
    for l = 1:k
        for z = A.rowptr[l]:A.rowptr[l+1]-1
            i = A.colind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[j,l]
            end
        end
    end
    return C
end

###

function _mul(B::Matrix{Tv}, A::SparseCSR{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, k = size(B)
    k, n = size(A)
    C = zeros(m,n)
    for l = 1:k
        for z = A.rowptr[l]:A.rowptr[l+1]-1
            j = A.colind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[i,l]
            end
        end
    end
    return C
end

function _mul(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::SparseCSR{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    k, m = size(B)
    k, n = size(A)
    C = zeros(m,n)
    for l = 1:k
        for z = A.rowptr[l]:A.rowptr[l+1]-1
            j = A.colind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[l,i]
            end
        end
    end
    return C
end

function _mul(B::Matrix{Tv}, At::Adjoint{SparseCSR{Tv,Ti}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    m, k = size(B)
    n, k = size(A)
    C = zeros(m,n)
    for j = 1:n
        for z = A.rowptr[j]:A.rowptr[j+1]-1
            l = A.colind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[i,l]
            end
        end
    end
    return C
end

function _mul(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, At::Adjoint{SparseCSR{Tv,Ti}})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    A = At.parent
    k, m = size(B)
    n, k = size(A)
    C = zeros(m,n)
    for j = 1:n
        for z = A.rowptr[j]:A.rowptr[j+1]-1
            l = A.colind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[l,i]
            end
        end
    end
    return C
end

#################### CSC

function _mul(A::SparseCSC{Tv,Ti}, x::Vector{Tv})::Vector{Tv} where {Tv, Ti}
    m, n = size(A)
    y = zeros(m)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            i = A.rowind[z]
            y[i] += A.val[z] * x[j]
        end
    end
    return y
end

function _mul(At::Adjoint{SparseCSC{Tv,Ti}}, x::Vector{Tv})::Vector{Tv} where {Tv, Ti}
    A = At.parent
    n, m = size(A)
    y = zeros(m)
    for i = 1:m
        for z = A.colptr[i]:A.colptr[i+1]-1
            j = A.rowind[z]
            y[i] += A.val[z] * x[j]
        end
    end
    return y
end

###

function _mul(A::SparseCSC{Tv,Ti}, B::Matrix{Tv})::Matrix{Tv} where {Tv, Ti}
    m, k = size(A)
    k, n = size(B)
    C = zeros(m,n)
    for l = 1:k
        for z = A.colptr[l]:A.colptr[l+1]-1
            i = A.rowind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[l,j]
            end
        end
    end
    return C
end

function _mul(A::SparseCSC{Tv,Ti}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    m, k = size(A)
    n, k = size(B)
    C = zeros(m,n)
    for l = 1:k
        for z = A.colptr[l]:A.colptr[l+1]-1
            i = A.rowind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[j,l]
            end
        end
    end
    return C
end

function _mul(At::Adjoint{SparseCSC{Tv,Ti}}, B::Matrix{Tv})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    k, m = size(A)
    k, n = size(B)
    C = zeros(m,n)
    for i = 1:m
        for z = A.colptr[i]:A.colptr[i+1]-1
            l = A.rowind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[l,j]
            end
        end
    end
    return C
end

function _mul(At::Adjoint{SparseCSC{Tv,Ti}}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    B = Bt.parent
    k, m = size(A)
    n, k = size(B)
    C = zeros(m,n)
    for i = 1:m
        for z = A.colptr[i]:A.colptr[i+1]-1
            l = A.rowind[z]
            for j = 1:n
                C[i,j] += A.val[z] * B[j,l]
            end
        end
    end
    return C
end

###

function _mul(B::Matrix{Tv}, A::SparseCSC{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, k = size(B)
    k, n = size(A)
    C = zeros(m,n)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            l = A.rowind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[i,l]
            end
        end
    end
    return C
end

function _mul(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::SparseCSC{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    k, m = size(B)
    k, n = size(A)
    C = zeros(m,n)
    for j = 1:n
        for z = A.colptr[j]:A.colptr[j+1]-1
            l = A.rowind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[l,i]
            end
        end
    end
    return C
end

function _mul(B::Matrix{Tv}, At::Adjoint{SparseCSC{Tv,Ti}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    m, k = size(B)
    n, k = size(A)
    C = zeros(m,n)
    for l = 1:k
        for z = A.colptr[l]:A.colptr[l+1]-1
            j = A.rowind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[i,l]
            end
        end
    end
    return C
end

function _mul(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, At::Adjoint{SparseCSC{Tv,Ti}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    B = Bt.parent
    k, m = size(B)
    n, k = size(A)
    C = zeros(m,n)
    for l = 1:k
        for z = A.colptr[l]:A.colptr[l+1]-1
            j = A.rowind[z]
            for i = 1:m
                C[i,j] += A.val[z] * B[l,i]
            end
        end
    end
    return C
end

#################### COO

function _mul(A::SparseCOO{Tv,Ti}, x::Vector{Tv})::Vector{Tv} where {Tv, Ti}
    m, n = size(A)
    y = zeros(m)
    for z = 1:nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        y[i] += A.val[z] * x[j]
    end
    return y
end

function _mul(At::Adjoint{SparseCOO{Tv,Ti}}, x::Vector{Tv})::Vector{Tv} where {Tv, Ti}
    A = At.parent
    n, m = size(A)
    y = zeros(m)
    for z = 1:nnz(A)
        j = A.rowind[z]
        i = A.colind[z]
        y[i] += A.val[z] * x[j]
    end
    return y
end

###

function _mul(A::SparseCOO{Tv,Ti}, B::Matrix{Tv})::Matrix{Tv} where {Tv, Ti}
    m, k = size(A)
    k, n = size(B)
    C = zeros(m,n)
    for z = 1:nnz(A)
        i = A.rowind[z]
        l = A.colind[z]
        for j = 1:n
            C[i,j] += A.val[z] * B[l,j]
        end
    end
    return C
end

function _mul(A::SparseCOO{Tv,Ti}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    m, k = size(A)
    n, k = size(B)
    C = zeros(m,n)
    for z = 1:nnz(A)
        i = A.rowind[z]
        l = A.colind[z]
        for j = 1:n
            C[i,j] += A.val[z] * B[j,l]
        end
    end
    return C
end

function _mul(At::Adjoint{SparseCOO{Tv,Ti}}, B::Matrix{Tv})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    k, m = size(A)
    k, n = size(B)
    C = zeros(m,n)
    for z = 1:nnz(A)
        l = A.rowind[z]
        i = A.colind[z]
        for j = 1:n
            C[i,j] += A.val[z] * B[l,j]
        end
    end
    return C
end

function _mul(At::Adjoint{SparseCOO{Tv,Ti}}, Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    B = Bt.parent
    k, m = size(A)
    n, k = size(B)
    C = zeros(m,n)
    for z = 1:nnz(A)
        l = A.rowind[z]
        i = A.colind[z]
        for j = 1:n
            C[i,j] += A.val[z] * B[j,l]
        end
    end
    return C
end

###

function _mul(B::Matrix{Tv}, A::SparseCOO{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    m, k = size(B)
    k, n = size(A)
    C = zeros(m,n)
    for z = 1:nnz(A)
        l = A.rowind[z]
        j = A.colind[z]
        for i = 1:m
            C[i,j] += A.val[z] * B[i,l]
        end
    end
    return C
end

function _mul(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::SparseCOO{Tv,Ti})::Matrix{Tv} where {Tv, Ti}
    B = Bt.parent
    k, m = size(B)
    k, n = size(A)
    C = zeros(m,n)
    for z = 1:nnz(A)
        l = A.rowind[z]
        j = A.colind[z]
        for i = 1:m
            C[i,j] += A.val[z] * B[l,i]
        end
    end
    return C
end

function _mul(B::Matrix{Tv}, At::Adjoint{SparseCOO{Tv,Ti}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    m, k = size(B)
    n, k = size(A)
    C = zeros(m,n)
    for z = 1:nnz(A)
        j = A.rowind[z]
        l = A.colind[z]
        for i = 1:m
            C[i,j] += A.val[z] * B[i,l]
        end
    end
    return C
end

function _mul(Bt::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, At::Adjoint{SparseCOO{Tv,Ti}})::Matrix{Tv} where {Tv, Ti}
    A = At.parent
    B = Bt.parent
    k, m = size(B)
    n, k = size(A)
    C = zeros(m,n)
    for z = 1:nnz(A)
        j = A.rowind[z]
        l = A.colind[z]
        for i = 1:m
            C[i,j] += A.val[z] * B[l,i]
        end
    end
    return C
end

######## override

function Base.:*(x::Tv, A::AbstractSparseM{Tv,Ti}) where {Tv<:Number,Ti}
    B = copy(A)
    B.val .*= x
    return B
end

function Base.:*(A::AbstractSparseM{Tv,Ti}, x::Tv) where {Tv<:Number,Ti}
    B = copy(A)
    B.val .*= x
    return B
end

function Base.:/(A::AbstractSparseM{Tv,Ti}, x::Tv) where {Tv<:Number,Ti}
    B = copy(A)
    B.val ./= x
    return B
end

(Base.:*)(A::SparseCSR{Tv,Ti}, x::Vector{Tv}) where {Tv,Ti} = _mul(A, x)
(Base.:*)(A::Adjoint{SparseCSR{Tv,Ti}}, x::Vector{Tv}) where {Tv,Ti} = _mul(A, x)

(Base.:*)(A::SparseCSR{Tv,Ti}, B::Matrix{Tv}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::SparseCSR{Tv,Ti}, B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::Adjoint{SparseCSR{Tv,Ti}}, B::Matrix{Tv}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::Adjoint{SparseCSR{Tv,Ti}}, B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = _mul(A, B)

(Base.:*)(B::Matrix{Tv}, A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::SparseCSR{Tv,Ti}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::Matrix{Tv}, A::Adjoint{SparseCSR{Tv,Ti}}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::Adjoint{SparseCSR{Tv,Ti}}) where {Tv,Ti} = _mul(B, A)

###

(Base.:*)(A::SparseCSC{Tv,Ti}, x::Vector{Tv}) where {Tv,Ti} = _mul(A, x)
(Base.:*)(A::Adjoint{SparseCSC{Tv,Ti}}, x::Vector{Tv}) where {Tv,Ti} = _mul(A, x)

(Base.:*)(A::SparseCSC{Tv,Ti}, B::Matrix{Tv}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::SparseCSC{Tv,Ti}, B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::Adjoint{SparseCSC{Tv,Ti}}, B::Matrix{Tv}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::Adjoint{SparseCSC{Tv,Ti}}, B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = _mul(A, B)

(Base.:*)(B::Matrix{Tv}, A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::SparseCSC{Tv,Ti}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::Matrix{Tv}, A::Adjoint{SparseCSC{Tv,Ti}}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::Adjoint{SparseCSC{Tv,Ti}}) where {Tv,Ti} = _mul(B, A)

###

(Base.:*)(A::SparseCOO{Tv,Ti}, x::Vector{Tv}) where {Tv,Ti} = _mul(A, x)
(Base.:*)(A::Adjoint{SparseCOO{Tv,Ti}}, x::Vector{Tv}) where {Tv,Ti} = _mul(A, x)

(Base.:*)(A::SparseCOO{Tv,Ti}, B::Matrix{Tv}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::SparseCOO{Tv,Ti}, B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::Adjoint{SparseCOO{Tv,Ti}}, B::Matrix{Tv}) where {Tv,Ti} = _mul(A, B)
(Base.:*)(A::Adjoint{SparseCOO{Tv,Ti}}, B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}) where {Tv,Ti} = _mul(A, B)

(Base.:*)(B::Matrix{Tv}, A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::SparseCOO{Tv,Ti}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::Matrix{Tv}, A::Adjoint{SparseCOO{Tv,Ti}}) where {Tv,Ti} = _mul(B, A)
(Base.:*)(B::LinearAlgebra.Adjoint{Tv,Matrix{Tv}}, A::Adjoint{SparseCOO{Tv,Ti}}) where {Tv,Ti} = _mul(B, A)

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

#################### sprand

function sprand(m, n, p::AbstractFloat, ::Type{T}) where {T}
    T(SparseArrays.sprand(m, n, p))
end

function sprandn(m, n, p::AbstractFloat, ::Type{T}) where {T}
    T(SparseArrays.sprandn(m, n, p))
end
