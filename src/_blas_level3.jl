## overload LinearAlgebra.BLAS.gemm!

# original interface

# function gemm!(transA::AbstractChar, transB::AbstractChar,
#     alpha::Union{($elty), Bool},
#     A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty},
#     beta::Union{($elty), Bool},
#     C::AbstractVecOrMat{$elty})

# blas level 3

import LinearAlgebra.BLAS: gemm!

for Tv in [:Float64]
    @eval begin

        ### CSR

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, A::SparseCSR{$Tv,Ti}, B::Matrix{$Tv},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(A)
                k, n = size(B)
                @. C *= beta
                for i = 1:m
                    for z = A.rowptr[i]:A.rowptr[i+1]-1
                        l = A.colind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[l,j]
                        end
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(A)
                n, k = size(B)
                @. C *= beta
                for i = 1:m
                    for z = A.rowptr[i]:A.rowptr[i+1]-1
                        l = A.colind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[j,l]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(A)
                k, n = size(B)
                @. C *= beta
                for l = 1:k
                    for z = A.rowptr[l]:A.rowptr[l+1]-1
                        i = A.colind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[l,j]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(A)
                n, k = size(B)
                @. C *= beta
                for l = 1:k
                    for z = A.rowptr[l]:A.rowptr[l+1]-1
                        i = A.colind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[j,l]
                        end
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### CSR2

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, B::Matrix{$Tv}, A::SparseCSR{$Tv,Ti},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(B)
                k, n = size(A)
                @. C *= beta
                for l = 1:k
                    for z = A.rowptr[l]:A.rowptr[l+1]-1
                        j = A.colind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[i,l]
                        end
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(B)
                n, k = size(A)
                @. C *= beta
                for j = 1:n
                    for z = A.rowptr[j]:A.rowptr[j+1]-1
                        l = A.colind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[i,l]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(B)
                k, n = size(A)
                @. C *= beta
                for l = 1:k
                    for z = A.rowptr[l]:A.rowptr[l+1]-1
                        j = A.colind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[l,i]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(B)
                n, k = size(A)
                @. C *= beta
                for j = 1:n
                    for z = A.rowptr[j]:A.rowptr[j+1]-1
                        l = A.colind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[l,i]
                        end
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### CSC

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, A::SparseCSC{$Tv,Ti}, B::Matrix{$Tv},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(A)
                k, n = size(B)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        i = A.rowind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[l,j]
                        end
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(A)
                n, k = size(B)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        i = A.rowind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[j,l]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(A)
                k, n = size(B)
                @. C *= beta
                for i = 1:m
                    for z = A.colptr[i]:A.colptr[i+1]-1
                        l = A.rowind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[l,j]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(A)
                n, k = size(B)
                @. C *= beta
                for i = 1:m
                    for z = A.colptr[i]:A.colptr[i+1]-1
                        l = A.rowind[z]
                        for j = 1:n
                            C[i,j] += alpha * A.val[z] * B[j,l]
                        end
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### CSC2

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, B::Matrix{$Tv}, A::SparseCSC{$Tv,Ti},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(B)
                k, n = size(A)
                @. C *= beta
                for j = 1:n
                    for z = A.colptr[j]:A.colptr[j+1]-1
                        l = A.rowind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[i,l]
                        end
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(B)
                n, k = size(A)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        j = A.rowind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[i,l]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(B)
                k, n = size(A)
                @. C *= beta
                for j = 1:n
                    for z = A.colptr[j]:A.colptr[j+1]-1
                        l = A.rowind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[l,i]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(B)
                n, k = size(A)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        j = A.rowind[z]
                        for i = 1:m
                            C[i,j] += alpha * A.val[z] * B[l,i]
                        end
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### COO

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, A::SparseCOO{$Tv,Ti}, B::Matrix{$Tv},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(A)
                k, n = size(B)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    i = A.rowind[z]
                    l = A.colind[z]
                    for j = 1:n
                        C[i,j] += alpha * A.val[z] * B[l,j]
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(A)
                n, k = size(B)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    i = A.rowind[z]
                    l = A.colind[z]
                    for j = 1:n
                        C[i,j] += alpha * A.val[z] * B[j,l]
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(A)
                k, n = size(B)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    l = A.rowind[z]
                    i = A.colind[z]
                    for j = 1:n
                        C[i,j] += alpha * A.val[z] * B[l,j]
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(A)
                n, k = size(B)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    l = A.rowind[z]
                    i = A.colind[z]
                    for j = 1:n
                        C[i,j] += alpha * A.val[z] * B[j,l]
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### COO2

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, B::Matrix{$Tv}, A::SparseCOO{$Tv,Ti},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(B)
                k, n = size(A)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    l = A.rowind[z]
                    j = A.colind[z]
                    for i = 1:m
                        C[i,j] += alpha * A.val[z] * B[i,l]
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(B)
                n, k = size(A)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    j = A.rowind[z]
                    l = A.colind[z]
                    for i = 1:m
                        C[i,j] += alpha * A.val[z] * B[i,l]
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(B)
                k, n = size(A)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    l = A.rowind[z]
                    j = A.colind[z]
                    for i = 1:m
                        C[i,j] += alpha * A.val[z] * B[l,i]
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(B)
                n, k = size(A)
                @. C *= beta
                for z = 1:SparseArrays.nnz(A)
                    j = A.rowind[z]
                    l = A.colind[z]
                    for i = 1:m
                        C[i,j] += alpha * A.val[z] * B[l,i]
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### SparseArrays.SparseMatrixCSC{Tv,Ti}

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti}, B::Matrix{$Tv},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(A)
                k, n = size(B)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        i = A.rowval[z]
                        for j = 1:n
                            C[i,j] += alpha * A.nzval[z] * B[l,j]
                        end
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(A)
                n, k = size(B)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        i = A.rowval[z]
                        for j = 1:n
                            C[i,j] += alpha * A.nzval[z] * B[j,l]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(A)
                k, n = size(B)
                @. C *= beta
                for i = 1:m
                    for z = A.colptr[i]:A.colptr[i+1]-1
                        l = A.rowval[z]
                        for j = 1:n
                            C[i,j] += alpha * A.nzval[z] * B[l,j]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(A)
                n, k = size(B)
                @. C *= beta
                for i = 1:m
                    for z = A.colptr[i]:A.colptr[i+1]-1
                        l = A.rowval[z]
                        for j = 1:n
                            C[i,j] += alpha * A.nzval[z] * B[j,l]
                        end
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

        ### SparseArrays.SparseMatrixCSC{$Tv,Ti}2

        function gemm!(transA::AbstractChar, transB::AbstractChar,
            alpha::Union{$Tv,Bool}, B::Matrix{$Tv}, A::SparseArrays.SparseMatrixCSC{$Tv,Ti},
            beta::Union{$Tv,Bool}, C::Matrix{$Tv}) where {Ti}
            if transA == 'N' && transB == 'N'
                m, k = size(B)
                k, n = size(A)
                @. C *= beta
                for j = 1:n
                    for z = A.colptr[j]:A.colptr[j+1]-1
                        l = A.rowval[z]
                        for i = 1:m
                            C[i,j] += alpha * A.nzval[z] * B[i,l]
                        end
                    end
                end
                C
            elseif transA == 'N' && transB == 'T'
                m, k = size(B)
                n, k = size(A)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        j = A.rowval[z]
                        for i = 1:m
                            C[i,j] += alpha * A.nzval[z] * B[i,l]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'N'
                k, m = size(B)
                k, n = size(A)
                @. C *= beta
                for j = 1:n
                    for z = A.colptr[j]:A.colptr[j+1]-1
                        l = A.rowval[z]
                        for i = 1:m
                            C[i,j] += alpha * A.nzval[z] * B[l,i]
                        end
                    end
                end
                C
            elseif transA == 'T' && transB == 'T'
                k, m = size(B)
                n, k = size(A)
                @. C *= beta
                for l = 1:k
                    for z = A.colptr[l]:A.colptr[l+1]-1
                        j = A.rowval[z]
                        for i = 1:m
                            C[i,j] += alpha * A.nzval[z] * B[l,i]
                        end
                    end
                end
                C
            else
                throw(ErrorException("trans should be 'N' or 'T'"))
            end
        end

    end
end
