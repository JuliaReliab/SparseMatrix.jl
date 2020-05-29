
export sprand, sprandn

#################### sprand

function sprand(m, n, p::AbstractFloat, ::Type{T}) where {T}
    T(SparseArrays.sprand(m, n, p))
end

function sprandn(m, n, p::AbstractFloat, ::Type{T}) where {T}
    T(SparseArrays.sprandn(m, n, p))
end
