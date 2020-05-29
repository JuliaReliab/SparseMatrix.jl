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
