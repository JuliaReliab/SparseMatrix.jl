using SparseMatrix
using LinearAlgebra
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
        @test z ≈ d1 ≈ d2 ≈ d3
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
        AA = Matrix(sparse(block(X)))
        A1 = Matrix(A1)
        A2 = Matrix(A2)
        BB = zeros(size(A1) .+ size(A2))
        BB[1:size(A1)[1],1:size(A1)[2]] = A1
        BB[size(A1)[1]+1:end,size(A1)[2]+1:end] = A2
        @test AA == BB
    end
end