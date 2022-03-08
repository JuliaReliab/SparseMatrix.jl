#################### CSR

for mat in (:SparseCSR, :SparseCSC, :SparseCOO)
    for elty in (:Float64,)
        @eval begin
            (Base.:*)(A::$mat{$elty,Ti}, x::Vector{$elty}) where {Ti} = gemv!('N', true, A, x, false, zeros(size(A,1)))
            (Base.:*)(At::Adjoint{$mat{$elty,Ti}}, x::Vector{$elty}) where {Ti} = gemv!('T', true, At.parent, x, false, zeros(size(At.parent,2)))
        end

        @eval begin
            (Base.:*)(A::$mat{$elty,Ti}, B::Matrix{$elty}) where {Ti} = gemm!('N', 'N', true, A, B, false, zeros(size(A,1), size(B,2)))
            (Base.:*)(A::$mat{$elty,Ti}, Bt::LinearAlgebra.Adjoint{$elty,Matrix{$elty}}) where {Ti} = gemm!('N', 'T', true, A, Bt.parent, false, zeros(size(A,1), size(Bt.parent,1)))
            (Base.:*)(At::Adjoint{$mat{$elty,Ti}}, B::Matrix{$elty}) where {Ti} = gemm!('T', 'N', true, At.parent, B, false, zeros(size(At.parent,2), size(B,2)))
            (Base.:*)(At::Adjoint{$mat{$elty,Ti}}, Bt::LinearAlgebra.Adjoint{$elty,Matrix{$elty}}) where {Ti} = gemm!('T', 'T', true, At.parent, Bt.parent, false, zeros(size(At.parent,2), size(Bt.parent,1)))

            (Base.:*)(B::Matrix{$elty}, A::$mat{$elty,Ti}) where {Ti} = gemm!('N', 'N', true, B, A, false, zeros(size(B,1), size(A,2)))
            (Base.:*)(Bt::LinearAlgebra.Adjoint{$elty,Matrix{$elty}}, A::$mat{$elty,Ti}) where {Ti} = gemm!('T', 'N', true, Bt.parent, A, false, zeros(size(Bt.parent,2), size(A,2)))
            (Base.:*)(B::Matrix{$elty}, At::Adjoint{$mat{$elty,Ti}}) where {Ti} = gemm!('N', 'T', true, B, At.parent, false, zeros(size(B,1), size(At.parent,1)))
            (Base.:*)(Bt::LinearAlgebra.Adjoint{$elty,Matrix{$elty}}, At::Adjoint{$mat{$elty,Ti}}) where {Ti} = gemm!('T', 'T', true, Bt.parent, At.parent, false, zeros(size(Bt.parent,2), size(At.parent,1)))
        end

        @eval begin
            (Base.:*)(x::Union{$elty,Bool}, A::$mat{$elty,Ti}) where {Ti} = scal!(x, copy(A))
            (Base.:*)(A::$mat{$elty,Ti}, x::Union{$elty,Bool}) where {Ti} = scal!(x, copy(A))
            (Base.:/)(A::$mat{$elty,Ti}, x::$elty) where {Ti} = scal!(Base.one($elty)/x, copy(A))

            (Base.:*)(x::Union{$elty,Bool}, At::Adjoint{$mat{$elty,Ti}}) where {Ti} = scal!(x, copy(At.parent))'
            (Base.:*)(At::Adjoint{$mat{$elty,Ti}}, x::Union{$elty,Bool}) where {Ti} = scal!(x, copy(At.parent))'
            (Base.:/)(At::Adjoint{$mat{$elty,Ti}}, x::$elty) where {Ti} = scal!(Base.one($elty)/x, copy(At.parent))'
        end
    end
end
