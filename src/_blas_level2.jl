## overload LinearAlgebra.BLAS.gemv!

# blas level 2

import LinearAlgebra.BLAS: gemv!

for Tv in [:Float64]
    @eval begin

        ## SparseArrays.SparseMatrixCSC
        function gemv!(trans::AbstractChar, alpha::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}, X::AbstractVector{$Tv},
            beta::Union{$Tv,Bool}, Y::AbstractVector{$Tv}) where {Ti}
            if trans == 'N'
                m, n = size(A)
                @. Y *= beta
                for j = 1:n
                    for z = A.colptr[j]:A.colptr[j+1]-1
                        i = A.rowval[z]
                        Y[i] += alpha * A.nzval[z] * X[j]
                    end
                end
                Y
            elseif trans == 'T'
                n, m = size(A)
                @. Y *= beta
                for i = 1:m
                    for z = A.colptr[i]:A.colptr[i+1]-1
                        j = A.rowval[z]
                        Y[i] += alpha * A.nzval[z] * X[j]
                    end
                end
                Y
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ## CSR
        function gemv!(trans::AbstractChar, alpha::Union{$Tv,Bool}, A::SparseCSR{$Tv,Ti}, X::AbstractVector{$Tv},
            beta::Union{$Tv,Bool}, Y::AbstractVector{$Tv}) where {Ti}
            if trans == 'N'
                m, n = size(A)
                @. Y *= beta
                for i = 1:m
                    for z = A.rowptr[i]:A.rowptr[i+1]-1
                        j = A.colind[z]
                        Y[i] += alpha * A.val[z] * X[j]
                    end
                end
                Y
            elseif trans == 'T'
                n, m = size(A)
                @. Y *= beta
                for j = 1:n
                    for z = A.rowptr[j]:A.rowptr[j+1]-1
                        i = A.colind[z]
                        Y[i] += alpha * A.val[z] * X[j]
                    end
                end
                Y
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ## CSC
        function gemv!(trans::AbstractChar, alpha::Union{$Tv,Bool}, A::SparseCSC{$Tv,Ti}, X::AbstractVector{$Tv},
            beta::Union{$Tv,Bool}, Y::AbstractVector{$Tv}) where {Ti}
            if trans == 'N'
                m, n = size(A)
                @. Y *= beta
                for j = 1:n
                    for z = A.colptr[j]:A.colptr[j+1]-1
                        i = A.rowind[z]
                        Y[i] += alpha * A.val[z] * X[j]
                    end
                end
                Y
            elseif trans == 'T'
                n, m = size(A)
                @. Y *= beta
                for i = 1:m
                    for z = A.colptr[i]:A.colptr[i+1]-1
                        j = A.rowind[z]
                        Y[i] += alpha * A.val[z] * X[j]
                    end
                end
                Y
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ## COO
        function gemv!(trans::AbstractChar, alpha::Union{$Tv,Bool}, A::SparseCOO{$Tv,Ti}, X::AbstractVector{$Tv},
            beta::Union{$Tv,Bool}, Y::AbstractVector{$Tv}) where {Ti}
            if trans == 'N'
                m, n = size(A)
                @. Y *= beta
                for z = 1:SparseArrays.nnz(A)
                    i = A.rowind[z]
                    j = A.colind[z]
                    Y[i] += alpha * A.val[z] * X[j]
                end
                Y
            elseif trans == 'T'
                n, m = size(A)
                @. Y *= beta
                for z = 1:SparseArrays.nnz(A)
                    j = A.rowind[z]
                    i = A.colind[z]
                    Y[i] += alpha * A.val[z] * X[j]
                end
                Y
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

    end

    @eval begin
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::Matrix{$Tv}) where {Ti}
            m, n = size(A)
            @. A *= beta
            for j = 1:n
                for i = 1:m
                    A[i,j] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## SparseArrays.SparseMatrixCSC
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.nzval *= beta
            for j = 1:n
                for z = A.colptr[j]:A.colptr[j+1]-1
                    i = A.rowval[z]
                    A.nzval[z] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## CSR
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseCSR{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.val *= beta
            for i = 1:m
                for z = A.rowptr[i]:A.rowptr[i+1]-1
                    j = A.colind[z]
                    A.val[z] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## CSC
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseCSC{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.val *= beta
            for j = 1:n
                for z = A.colptr[j]:A.colptr[j+1]-1
                    i = A.rowind[z]
                    A.val[z] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## COO
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseCSC{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.val *= beta
            for z = 1:SparseArrays.nnz(A)
                i = A.rowind[z]
                j = A.colind[z]
                A.val[z] += alpha * X[i] * Y[j]
            end
            A
        end
    end

    @eval begin
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::Matrix{$Tv}) where {Ti}
            m, n = size(A)
            @. A *= beta
            for j = 1:n
                for i = 1:m
                    A[i,j] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## SparseArrays.SparseMatrixCSC
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.nzval *= beta
            for j = 1:n
                for z = A.colptr[j]:A.colptr[j+1]-1
                    i = A.rowval[z]
                    A.nzval[z] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## CSR
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseCSR{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.val *= beta
            for i = 1:m
                for z = A.rowptr[i]:A.rowptr[i+1]-1
                    j = A.colind[z]
                    A.val[z] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## CSC
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseCSC{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.val *= beta
            for j = 1:n
                for z = A.colptr[j]:A.colptr[j+1]-1
                    i = A.rowind[z]
                    A.val[z] += alpha * X[i] * Y[j]
                end
            end
            A
        end

        ## COO
        function spger!(alpha::Union{$Tv,Bool}, X::AbstractVector{$Tv}, Y::AbstractVector{$Tv}, beta::Union{$Tv,Bool}, A::SparseCSC{$Tv,Ti}) where {Ti}
            m, n = size(A)
            @. A.val *= beta
            for z = 1:SparseArrays.nnz(A)
                i = A.rowind[z]
                j = A.colind[z]
                A.val[z] += alpha * X[i] * Y[j]
            end
            A
        end
    end
end
