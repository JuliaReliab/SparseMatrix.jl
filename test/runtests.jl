using SparseMatrix
using LinearAlgebra
using LinearAlgebra.BLAS
using SparseArrays
using Test

@testset "nnz1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x1=sparse(A)
        x2=SparseCSR(A)
        x3=SparseCSC(A)
        x4=SparseCOO(A)
        @test nnz(x1) == nnz(x2) == nnz(x3) == nnz(x4)
    end
end

@testset "tocsc1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x0 = sparse(A)
        x1=SparseCSR(A)
        x2=SparseCSC(A)
        x3=SparseCOO(A)
        y1=sparse(x1)
        y2=sparse(x2)
        y3=sparse(x3)
        if nnz(x1) != 0
            x1.val[1] = 0.0
            x2.val[1] = 0.0
            x3.val[1] = 0.0
        end
        @test x0 == y1 == y2 == y3
    end
end

@testset "multiple1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x=randn(n)
        C1 = A * x
        C2 = SparseCSR(A) * x
        C3 = SparseCSC(A) * x
        C4 = SparseCOO(A) * x
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple2" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(n,m,p))
        x=randn(n)
        C1 = A' * x
        C2 = SparseCSR(A)' * x
        C3 = SparseCSC(A)' * x
        C4 = SparseCOO(A)' * x
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple3" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,k,p))
        B=randn(k,n)
        C1 = A * B
        C2 = SparseCSR(A) * B
        C3 = SparseCSC(A) * B
        C4 = SparseCOO(A) * B
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple4" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(k,m,p))
        B=randn(k,n)
        C1 = A' * B
        C2 = SparseCSR(A)' * B
        C3 = SparseCSC(A)' * B
        C4 = SparseCOO(A)' * B
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple5" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,k,p))
        B=randn(n,k)
        C1 = A * B'
        C2 = SparseCSR(A) * B'
        C3 = SparseCSC(A) * B'
        C4 = SparseCOO(A) * B'
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple6" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(k,m,p))
        B=randn(n,k)
        C1 = A' * B'
        C2 = SparseCSR(A)' * B'
        C3 = SparseCSC(A)' * B'
        C4 = SparseCOO(A)' * B'
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple7" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=randn(m,k)
        B=Matrix(sprandn(k,n,p))
        C1 = A * B
        C2 = A * SparseCSR(B)
        C3 = A * SparseCSC(B)
        C4 = A * SparseCOO(B)
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple8" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=randn(k,m)
        B=Matrix(sprandn(k,n,p))
        C1 = A' * B
        C2 = A' * SparseCSR(B)
        C3 = A' * SparseCSC(B)
        C4 = A' * SparseCOO(B)
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple9" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=randn(m,k)
        B=Matrix(sprandn(n,k,p))
        C1 = A * B'
        C2 = A * SparseCSR(B)'
        C3 = A * SparseCSC(B)'
        C4 = A * SparseCOO(B)'
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "multiple10" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=randn(k,m)
        B=Matrix(sprandn(n,k,p))
        C1 = A' * B'
        C2 = A' * SparseCSR(B)'
        C3 = A' * SparseCSC(B)'
        C4 = A' * SparseCOO(B)'
        @test C1 ≈ C2 ≈ C3 ≈ C4
    end
end

@testset "diag1" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        d1 = spdiag(A)
        d2 = convert(Vector{Float64}, spdiag(SparseCSR(A)))
        d3 = convert(Vector{Float64}, spdiag(SparseCSC(A)))
        d4 = convert(Vector{Float64}, spdiag(SparseCOO(A)))
        @test d1 ≈ d2 ≈ d3 ≈ d4
    end
end

@testset "diag2" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        d1 = spdiag(A)
        d2 = spdiag(SparseCSR(A))
        d3 = spdiag(SparseCSC(A))
        d4 = spdiag(SparseCOO(A))
        for k = 1:length(d1)
            @test d1[k] ≈ d2[k] ≈ d3[k] ≈ d4[k]
        end
    end
end

@testset "diag3" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        d1 = abs.(spdiag(A))
        d2 = abs.(spdiag(SparseCSR(A)))
        d3 = abs.(spdiag(SparseCSC(A)))
        d4 = abs.(spdiag(SparseCOO(A)))
        @test d1 ≈ d2 ≈ d3 ≈ d4
    end
end

@testset "diag4" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        d1 = maximum(spdiag(A))
        d2 = maximum(spdiag(SparseCSR(A)))
        d3 = maximum(spdiag(SparseCSC(A)))
        d4 = maximum(spdiag(SparseCOO(A)))
        @test d1 ≈ d2 ≈ d3 ≈ d4
    end
end

@testset "copy" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        d1 = Matrix(copy(SparseCSR(A)))
        d2 = Matrix(copy(SparseCSC(A)))
        d3 = Matrix(copy(SparseCOO(A)))
        @test A ≈ d1 ≈ d2 ≈ d3
    end
end

@testset "zero" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        z = zero(A)
        d1 = Matrix(zero(SparseCSR(A)))
        d2 = Matrix(zero(SparseCSC(A)))
        d3 = Matrix(zero(SparseCOO(A)))
        d4 = Matrix(zero(SparseArrays.SparseMatrixCSC(A)))
        @test z ≈ d1 ≈ d2 ≈ d3 ≈ d4
    end
end

@testset "linearlize" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = SparseCSC(sprandn(m,n,p))
        B = copy(A)
        for i = 1:length(B)
            B[i] *= 100.0
        end
        @test A.val * 100.0 ≈ B.val
        A = SparseCSC(sprandn(m,n,p))
        B = copy(A)
        for i = eachindex(B)
            B[i] *= 100.0
        end
        @test A.val * 100.0 ≈ B.val
    end
end

@testset "block 1" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A1 = SparseCSC(sprandn(m,n,p))
        x1 = rand(n,1)
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A2 = SparseCSC(sprandn(m,n,p))
        x2 = rand(n,1)

        y0 = [A1 * x1; A2 * x2]

        X = BlockCOO(2,2, [(1,1,A1), (2,2,A2)])
        y = SparseCOO(X) * [x1; x2]

        @test Matrix(SparseCOO(X)) == Matrix(SparseCSR(X)) == Matrix(SparseCSC(X)) == Matrix(sparse(X))
        @test y0 ≈ y
    end
end

@testset "block 2" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A1 = SparseCSR(sprandn(m,n,p))
        x1 = rand(n,1)
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A2 = SparseCOO(sprandn(m,n,p))
        x2 = rand(n,1)

        y0 = [A1 * x1; A2 * x2]

        X = BlockCOO(2,2, [(1,1,A1), (2,2,A2)])
        y = SparseCOO(X) * [x1; x2]

        @test Matrix(SparseCOO(X)) == Matrix(SparseCSR(X)) == Matrix(SparseCSC(X)) == Matrix(sparse(X))
        @test y0 ≈ y
    end
end

@testset "block 3" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A1 = Matrix(sprandn(m,n,p))
        x1 = rand(n,1)
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A2 = SparseCSC(sprandn(m,n,p))
        x2 = rand(n,1)

        y0 = [A1 * x1; A2 * x2]

        X = BlockCOO(2,2, [(1,1,A1), (2,2,A2)])
        y = SparseCOO(X) * [x1; x2]

        @test Matrix(SparseCOO(X)) == Matrix(SparseCSR(X)) == Matrix(SparseCSC(X)) == Matrix(sparse(X))
        @test y0 ≈ y
    end
end

@testset "spdiag 1" begin
    for ite = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A = Matrix(sprandn(m,n,p))
        d0 = spdiag(A)
        d1 = spdiag(SparseCSR(A))
        d2 = spdiag(SparseCSC(A))
        d3 = spdiag(SparseCOO(A))
        d4 = spdiag(SparseMatrixCSC(A))
        v0 = [d0.val[i] for i = d0.index if !iszero(d0.val[i])]
        v1 = [d1.val[i] for i = d1.index if !iszero(i)]
        v2 = [d2.val[i] for i = d2.index if !iszero(i)]
        v3 = [d3.val[i] for i = d3.index if !iszero(i)]
        v4 = [d4.val[i] for i = d4.index if !iszero(i)]
        @test v0 == v1 == v2 == v3 == v4
    end
end

@testset "block 4" begin
    for i = 1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A1 = Matrix(sprandn(m,n,p))
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A2 = SparseCSC(sprandn(m,n,p))
        X = spzeros(AbstractMatrix{Float64}, 2, 2)
        X[1,1] = A1
        X[2,2] = A2
        if sum(A1 .!= 0) != 0 && SparseArrays.nnz(A2) != 0
            AA = Matrix(sparse(block(X)))
            A1 = Matrix(A1)
            A2 = Matrix(A2)
            BB = zeros(size(A1) .+ size(A2))
            BB[1:size(A1)[1],1:size(A1)[2]] = A1
            BB[size(A1)[1]+1:end,size(A1)[2]+1:end] = A2
            @test AA == BB
        end
    end
end

@testset "gemvN" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x=randn(n)
        y=randn(m)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * A * x + beta * y
        z2 = alpha * SparseCSR(A) * x + beta * y
        z3 = alpha * SparseCSC(A) * x + beta * y
        z4 = alpha * SparseCOO(A) * x + beta * y
        z5 = alpha * sparse(A) * x + beta * y
        C1 = gemv!('N', alpha, A, x, beta, copy(y))
        C2 = gemv!('N', alpha, SparseCSR(A), x, beta, copy(y))
        C3 = gemv!('N', alpha, SparseCSC(A), x, beta, copy(y))
        C4 = gemv!('N', alpha, SparseCOO(A), x, beta, copy(y))
        C5 = gemv!('N', alpha, sparse(A), x, beta, copy(y))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemvT" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(n,m,p))
        x=randn(n)
        y=randn(m)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * A' * x + beta * y
        z2 = alpha * SparseCSR(A)' * x + beta * y
        z3 = alpha * SparseCSC(A)' * x + beta * y
        z4 = alpha * SparseCOO(A)' * x + beta * y
        z5 = alpha * sparse(A)' * x + beta * y
        C1 = gemv!('T', alpha, A, x, beta, copy(y))
        C2 = gemv!('T', alpha, SparseCSR(A), x, beta, copy(y))
        C3 = gemv!('T', alpha, SparseCSC(A), x, beta, copy(y))
        C4 = gemv!('T', alpha, SparseCOO(A), x, beta, copy(y))
        C5 = gemv!('T', alpha, sparse(A), x, beta, copy(y))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmNN1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,k,p))
        B=randn(k,n)
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * A * B + beta * C
        z2 = alpha * SparseCSR(A) * B + beta * C
        z3 = alpha * SparseCSC(A) * B + beta * C
        z4 = alpha * SparseCOO(A) * B + beta * C
        z5 = alpha * sparse(A) * B + beta * C
        C1 = gemm!('N', 'N', alpha, A, B, beta, copy(C))
        C2 = gemm!('N', 'N', alpha, SparseCSR(A), B, beta, copy(C))
        C3 = gemm!('N', 'N', alpha, SparseCSC(A), B, beta, copy(C))
        C4 = gemm!('N', 'N', alpha, SparseCOO(A), B, beta, copy(C))
        C5 = gemm!('N', 'N', alpha, sparse(A), B, beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmNN2" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        B=randn(m,k)
        A=Matrix(sprandn(k,n,p))
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * B * A + beta * C
        z2 = alpha * B * SparseCSR(A) + beta * C
        z3 = alpha * B * SparseCSC(A) + beta * C
        z4 = alpha * B * SparseCOO(A) + beta * C
        z5 = alpha * B * sparse(A) + beta * C
        C1 = gemm!('N', 'N', alpha, B, A, beta, copy(C))
        C2 = gemm!('N', 'N', alpha, B, SparseCSR(A), beta, copy(C))
        C3 = gemm!('N', 'N', alpha, B, SparseCSC(A), beta, copy(C))
        C4 = gemm!('N', 'N', alpha, B, SparseCOO(A), beta, copy(C))
        C5 = gemm!('N', 'N', alpha, B, sparse(A), beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmNT1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,k,p))
        B=randn(n,k)
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * A * B' + beta * C
        z2 = alpha * SparseCSR(A) * B' + beta * C
        z3 = alpha * SparseCSC(A) * B' + beta * C
        z4 = alpha * SparseCOO(A) * B' + beta * C
        z5 = alpha * sparse(A) * B' + beta * C
        C1 = gemm!('N', 'T', alpha, A, B, beta, copy(C))
        C2 = gemm!('N', 'T', alpha, SparseCSR(A), B, beta, copy(C))
        C3 = gemm!('N', 'T', alpha, SparseCSC(A), B, beta, copy(C))
        C4 = gemm!('N', 'T', alpha, SparseCOO(A), B, beta, copy(C))
        C5 = gemm!('N', 'T', alpha, sparse(A), B, beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmNT2" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        B=randn(m,k)
        A=Matrix(sprandn(n,k,p))
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * B * A' + beta * C
        z2 = alpha * B * SparseCSR(A)' + beta * C
        z3 = alpha * B * SparseCSC(A)' + beta * C
        z4 = alpha * B * SparseCOO(A)' + beta * C
        z5 = alpha * B * sparse(A)' + beta * C
        C1 = gemm!('N', 'T', alpha, B, A, beta, copy(C))
        C2 = gemm!('N', 'T', alpha, B, SparseCSR(A), beta, copy(C))
        C3 = gemm!('N', 'T', alpha, B, SparseCSC(A), beta, copy(C))
        C4 = gemm!('N', 'T', alpha, B, SparseCOO(A), beta, copy(C))
        C5 = gemm!('N', 'T', alpha, B, sparse(A), beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmTN1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(k,m,p))
        B=randn(k,n)
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * A' * B + beta * C
        z2 = alpha * SparseCSR(A)' * B + beta * C
        z3 = alpha * SparseCSC(A)' * B + beta * C
        z4 = alpha * SparseCOO(A)' * B + beta * C
        z5 = alpha * sparse(A)' * B + beta * C
        C1 = gemm!('T', 'N', alpha, A, B, beta, copy(C))
        C2 = gemm!('T', 'N', alpha, SparseCSR(A), B, beta, copy(C))
        C3 = gemm!('T', 'N', alpha, SparseCSC(A), B, beta, copy(C))
        C4 = gemm!('T', 'N', alpha, SparseCOO(A), B, beta, copy(C))
        C5 = gemm!('T', 'N', alpha, sparse(A), B, beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmTN2" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        B=randn(k,m)
        A=Matrix(sprandn(k,n,p))
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * B' * A + beta * C
        z2 = alpha * B' * SparseCSR(A) + beta * C
        z3 = alpha * B' * SparseCSC(A) + beta * C
        z4 = alpha * B' * SparseCOO(A) + beta * C
        z5 = alpha * B' * sparse(A) + beta * C
        C1 = gemm!('T', 'N', alpha, B, A, beta, copy(C))
        C2 = gemm!('T', 'N', alpha, B, SparseCSR(A), beta, copy(C))
        C3 = gemm!('T', 'N', alpha, B, SparseCSC(A), beta, copy(C))
        C4 = gemm!('T', 'N', alpha, B, SparseCOO(A), beta, copy(C))
        C5 = gemm!('T', 'N', alpha, B, sparse(A), beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmTT1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        A=Matrix(sprandn(k,m,p))
        B=randn(n,k)
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * A' * B' + beta * C
        z2 = alpha * SparseCSR(A)' * B' + beta * C
        z3 = alpha * SparseCSC(A)' * B' + beta * C
        z4 = alpha * SparseCOO(A)' * B' + beta * C
        z5 = alpha * sparse(A)' * B' + beta * C
        C1 = gemm!('T', 'T', alpha, A, B, beta, copy(C))
        C2 = gemm!('T', 'T', alpha, SparseCSR(A), B, beta, copy(C))
        C3 = gemm!('T', 'T', alpha, SparseCSC(A), B, beta, copy(C))
        C4 = gemm!('T', 'T', alpha, SparseCOO(A), B, beta, copy(C))
        C5 = gemm!('T', 'T', alpha, sparse(A), B, beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "gemmTT2" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        k = rand(1:20)
        p = rand()
        B=randn(k,m)
        A=Matrix(sprandn(n,k,p))
        C=randn(m,n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        z1 = alpha * B' * A' + beta * C
        z2 = alpha * B' * SparseCSR(A)' + beta * C
        z3 = alpha * B' * SparseCSC(A)' + beta * C
        z4 = alpha * B' * SparseCOO(A)' + beta * C
        z5 = alpha * B' * sparse(A)' + beta * C
        C1 = gemm!('T', 'T', alpha, B, A, beta, copy(C))
        C2 = gemm!('T', 'T', alpha, B, SparseCSR(A), beta, copy(C))
        C3 = gemm!('T', 'T', alpha, B, SparseCSC(A), beta, copy(C))
        C4 = gemm!('T', 'T', alpha, B, SparseCOO(A), beta, copy(C))
        C5 = gemm!('T', 'T', alpha, B, sparse(A), beta, copy(C))
        for x = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
            for y = [z1, z2, z3, z4, z5, C1, C2, C3, C4, C5]
                @test isapprox(x, y)
            end
        end
    end
end

@testset "spger" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x=randn(m)
        y=randn(n)
        alpha = randn(1)[1]
        beta = randn(1)[1]
        C1 = spger!(alpha, x, y, beta, copy(A)) .* A
        C2 = Matrix(spger!(alpha, x, y, beta, SparseCSR(A))) .* A
        C3 = Matrix(spger!(alpha, x, y, beta, SparseCSC(A))) .* A
        C4 = Matrix(spger!(alpha, x, y, beta, SparseCOO(A))) .* A
        C5 = Matrix(spger!(alpha, x, y, beta, sparse(A))) .* A
        for x = [C1, C2, C3, C4, C5]
            for y = [C1, C2, C3, C4, C5]
                @test isapprox(x,y)
            end
        end
    end
end

@testset "axpy" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        B=copy(A) * 10.0
        a= randn(1)[1]
        C1 = a * A + B
        C2 = Matrix(axpy!(a, SparseCSR(A), SparseCSR(B)))
        C3 = Matrix(axpy!(a, SparseCSC(A), SparseCSC(B)))
        C4 = Matrix(axpy!(a, SparseCOO(A), SparseCOO(B)))
        C5 = Matrix(axpy!(a, sparse(A), sparse(B)))
        for x = [C1, C2, C3, C4, C5]
            for y = [C1, C2, C3, C4, C5]
                @test isapprox(x,y)
            end
        end
    end
end

@testset "ell1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x0 = sparse(A)
        x1=SparseCSR(A)
        x2=SparseCSC(A)
        x3=SparseCOO(A)
        y0 = SparseELL1(x0)
        for i = 1:y0.m
            for z = 1:y0.k
                if y0.idx[i,z] != 0
                    j = y0.idx[i,z]
                    @test A[i,j] == y0.val[i,z]
                end
            end
        end
        y1 = SparseELL1(x1)
        for i = 1:y1.m
            for z = 1:y1.k
                if y1.idx[i,z] != 0
                    j = y1.idx[i,z]
                    @test A[i,j] == y1.val[i,z]
                end
            end
        end
        y2 = SparseELL1(x2)
        for i = 1:y2.m
            for z = 1:y2.k
                if y2.idx[i,z] != 0
                    j = y2.idx[i,z]
                    @test A[i,j] == y2.val[i,z]
                end
            end
        end
        y3 = SparseELL1(x3)
        for i = 1:y3.m
            for z = 1:y3.k
                if y3.idx[i,z] != 0
                    j = y3.idx[i,z]
                    @test A[i,j] == y3.val[i,z]
                end
            end
        end       
    end
end

@testset "ell2" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=Matrix(sprandn(m,n,p))
        x0 = sparse(A)
        x1=SparseCSR(A)
        x2=SparseCSC(A)
        x3=SparseCOO(A)
        y0 = SparseELL2(x0)
        for j = 1:y0.n
            for z = 1:y0.k
                if y0.idx[j,z] != 0
                    i = y0.idx[j,z]
                    @test A[i,j] == y0.val[j,z]
                end
            end
        end
        y1 = SparseELL2(x1)
        for j = 1:y1.n
            for z = 1:y1.k
                if y1.idx[j,z] != 0
                    i = y1.idx[j,z]
                    @test A[i,j] == y1.val[j,z]
                end
            end
        end
        y2 = SparseELL2(x2)
        for j = 1:y2.n
            for z = 1:y2.k
                if y2.idx[j,z] != 0
                    i = y2.idx[j,z]
                    @test A[i,j] == y2.val[j,z]
                end
            end
        end
        y3 = SparseELL2(x3)
        for j = 1:y3.n
            for z = 1:y3.k
                if y3.idx[j,z] != 0
                    i = y3.idx[j,z]
                    @test A[i,j] == y3.val[j,z]
                end
            end
        end
    end
end
