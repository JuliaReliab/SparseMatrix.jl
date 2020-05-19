using SparseMatrix
using LinearAlgebra
using Test

@testset "multiple1" begin
    for i=1:100
        m = rand(1:20)
        n = rand(1:20)
        p = rand()
        A=sprandn(m,n,p,Matrix)
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
        A=sprandn(n,m,p,Matrix)
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
        A=sprandn(m,k,p,Matrix)
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
        A=sprandn(k,m,p,Matrix)
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
        A=sprandn(m,k,p,Matrix)
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
        A=sprandn(k,m,p,Matrix)
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
        B=sprandn(k,n,p,Matrix)
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
        B=sprandn(k,n,p,Matrix)
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
        B=sprandn(n,k,p,Matrix)
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
        B=sprandn(n,k,p,Matrix)
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
        A = sprandn(m,n,p,Matrix)
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
        A = sprandn(m,n,p,Matrix)
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
        A = sprandn(m,n,p,Matrix)
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
        A = sprandn(m,n,p,Matrix)
        d1 = maximum(spdiag(A))
        d2 = maximum(spdiag(SparseCSR(A)))
        d3 = maximum(spdiag(SparseCSC(A)))
        d4 = maximum(spdiag(SparseCOO(A)))
        @test d1 ≈ d2 ≈ d3 ≈ d4
    end
end
