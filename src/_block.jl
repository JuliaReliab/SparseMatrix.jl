export BlockCOO, block

"""
    BlockCOO(Tv,Ti) <: AbstractSparseM{Tv,Ti}

Type that represents a block matrix

### Fileds
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `val::Vector{<:AbstractMatrix{Tv}}`: a vector of non-zero elements whose type is a subtype of AbstractMatrix{Tv}.
- `rowind::Vector{Ti}`: a vector indicating the row index for the corredponding element of `val`.
- `colind::Vector{Ti}`: a vector indicating the column index for the corredponding element of `val`.

### Notes

This is an essential same as SparseCOO{<:AbstractSparseM{Tv,Ti},Ti} in terms of structure.
This is used to distinguish from SparseCOO{<:AbstractSparseM{Tv,Ti},Ti} because the
ambiguous error occurs in SparseCSR/SparseCSC/SparseCOO functions when we want to
make a sparse matrix where the matrix elements are extracted.
"""
struct BlockCOO{Tv,Ti} <: AbstractSparseM{Tv,Ti}
    m::Ti
    n::Ti
    val::Vector{<:AbstractMatrix{Tv}}
    rowind::Vector{Ti}
    colind::Vector{Ti}
end

"""
    BlockCOO(m::Ti, n::Ti, elem::AbstractArray{Tuple{Ti,Ti,Tv},1}) where {Tv,Ti}

Create BlockCOO from an array of matrices.

### Input
- `m::Ti`: the number of rows whose type is Ti
- `n::Ti`: the number of columns whose type is Ti
- `elem::AbstractArray{Tuple{Ti,Ti,Tv},1}`: a vector of non-zero matrices.
"""
function BlockCOO(m::Ti, n::Ti, elem::AbstractArray{Tuple{Ti,Ti,Tv},1}) where {Tv,Ti}
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
    BlockCOO(m, n, val, rowind, colind)
end

function Base.show(io::IO, ::MIME"text/plain", A::BlockCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" Block-COO")
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "("*string(i)*","*string(j)*") "*string(A.val[z]))
    end
end

"""
     block(A)

Create a BlockCOO from a given matrix A. The matrix A is a matrix whose element is a value of AbstractMatrix.
"""
block(A::SparseCOO{<:AbstractMatrix{Tv},Ti}) where {Tv,Ti} = BlockCOO{Tv,Ti}(A.m, A.n, copy(A.val), copy(A.rowind), copy(A.colind))
block(A::SparseCSR{<:AbstractMatrix{Tv},Ti}) where {Tv,Ti} = block(SparseCOO(A))
block(A::SparseCSC{<:AbstractMatrix{Tv},Ti}) where {Tv,Ti} = block(SparseCOO(A))
block(A::SparseArrays.SparseMatrixCSC{<:AbstractMatrix{Tv},Ti}) where {Tv,Ti} = block(SparseCOO(A))

SparseCSR(A::BlockCOO{Tv,Ti}) where {Tv,Ti} = _tocsr(_tocoo(A))
SparseCSC(A::BlockCOO{Tv,Ti}) where {Tv,Ti} = _tocsc(_tocoo(A))
SparseCOO(A::BlockCOO{Tv,Ti}) where {Tv,Ti} = _tocoo(A)
SparseArrays.sparse(A::BlockCOO{Tv,Ti}) where {Tv,Ti} = SparseArrays.sparse(_tocoo(A))

function Base.show(io::IO, A::BlockCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    println(io, string(A.m)*"x"*string(A.n)*" COO-BlockMatrix")
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        println(io, "($(i), $(j)) $(A.val[z])")
    end
end

function _tocoo(A::BlockCOO{Tv,Ti}) where {Tv,Ti}
    m, n = size(A)
    bi = zeros(Ti, m)
    bj = zeros(Ti, n)
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        m, n = size(A.val[z])
        if bi[i] == Ti(0)
            bi[i] = m
        else
            @assert bi[i] == m "assert error ($i, $j) ($m, $n)"
        end
        if bj[j] == Ti(0)
            bj[j] = n
        else
            @assert bj[j] == n "assert error ($i, $j) ($m, $n)"
        end
    end
    @assert all(bi .!= Ti(0))
    @assert all(bj .!= Ti(0))

    ## make
    m, n = sum(bi), sum(bj)
    ci = zero(bi)
    ci[1] = Ti(0)
    for i = 1:length(bi)-1
        ci[i+1] = ci[i] + bi[i]
    end
    cj = zero(bj)
    cj[1] = Ti(0)
    for j = 1:length(bj)-1
        cj[j+1] = cj[j] + bj[j]
    end
    val = Tv[]
    rowind = Ti[]
    colind = Ti[]
    for z = 1:SparseArrays.nnz(A)
        i = A.rowind[z]
        j = A.colind[z]
        coo = SparseCOO(A.val[z])
        append!(val, coo.val)
        append!(rowind, coo.rowind .+ ci[i])
        append!(colind, coo.colind .+ cj[j])
    end
    SparseCOO(m, n, val, rowind, colind)
end