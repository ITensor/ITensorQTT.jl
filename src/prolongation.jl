# https://arxiv.org/abs/1802.07259
function prolongation_tensor()
    P = zeros(2, 2, 2, 2)
    P[1, 1, 1, 1] = 1.0
    P[1, 1, 2, 2] = 1.0
    P[1, 2, 2, 1] = 1.0
    P[2, 2, 1, 2] = 1.0
    return P
end

# https://arxiv.org/abs/1802.07259
function prolongation_mpo(s::Vector{<:Index})
    L = length(s) - 1
    l = [Index(2, "l=$(j)↔$(j+1)") for j = 0:L]
    P⁰ = onehot(l[1] => 1)
    P⃗ = [itensor(prolongation_tensor(), l[j], l[j+1], s[j], s[j]') for j = 1:L]
    P⃗[1] *= P⁰
    Pᴸ⁺¹ = ITensor(l[L+1], s[L+1]')
    Pᴸ⁺¹[1, 1] = 1.0
    Pᴸ⁺¹[1, 2] = 0.5
    Pᴸ⁺¹[2, 2] = 0.5
    return MPO([P⃗; Pᴸ⁺¹])
end

function prolongation_mpo(s::Tuple{Vararg{Vector{<:Index}}})
    return interleave(prolongation_mpo.(s)...)
end

function prolongate(ψ::MPS, s::Tuple{Vararg{Index}}; cutoff = 1e-8, kwargs...)
    ndims = length(s)
    s_original = siteinds_per_dimension(Val(length(s)), ψ)
    s = vcat.(s_original, s)
    P = prolongation_mpo(s)
    ψ̃ = [ψ; MPS(fill(ITensor(1.0), ndims))]

    # Dummy index to make `apply` work
    s_end = ntuple(_ -> Index(1, "Site"), ndims)
    p = onehot.(s_end .=> 1)
    for j = 1:ndims
        ψ̃[end-ndims+j] *= p[j]
        P[end-ndims+j] *= p[j]
    end

    return apply(P, ψ̃; cutoff, kwargs...)
end

function prolongate(ψ::MPS, s::Tuple{Vararg{Vector{<:Index}}}; cutoff = 1e-8, kwargs...)
    for sⱼ in zip(s...)
        ψ = prolongate(ψ, sⱼ; cutoff, kwargs...)
    end
    return ψ
end

function prolongate(ψ::MPS, s::Index; cutoff = 1e-8, kwargs...)
    return prolongate(ψ, (s,); cutoff, kwargs...)
end

function prolongate(ψ::MPS, s::Vector{<:Index}; cutoff = 1e-8, kwargs...)
    return prolongate(ψ, (s,); cutoff, kwargs...)
end

#
# Retraction
#

function retraction_mpo(s::Tuple{Vararg{Vector{<:Index}}})
    return swapprime(prolongation_mpo(s), 0 => 1)
end

function retract_one_site(::Val{ndims}, ψ::MPS; cutoff = 1e-8, kwargs...) where {ndims}
    n = length(ψ)
    s = siteinds_per_dimension(Val(ndims), ψ)
    # Dummy indices to make `apply` work
    d = [Index(1, "dummy") for j = 1:ndims]
    R = retraction_mpo(s)
    for j = 1:ndims
        R[length(ψ)-ndims+j] *= onehot(d[j] => 1)
    end
    Rψ = apply(R, ψ; cutoff, kwargs...)
    for j = 1:ndims
        Rψ[n-ndims+j] *= onehot(dag(d[j]) => 1)
    end
    for j = 1:ndims
        Rψ[n-j] *= Rψ[n-j+1]
    end
    for j = 1:ndims
        pop!(Rψ)
    end
    return Rψ / (2^ndims)
end

function retract(ψ::MPS, nretract::Tuple{Vararg{Integer}}; cutoff = 1e-8, kwargs...)
    @assert all(==(first(nretract)), nretract)
    for j = 1:first(nretract)
        ψ = retract_one_site(Val(length(nretract)), ψ; cutoff, kwargs...)
    end
    return ψ
end

function retract(ψ::MPS, nretract::Integer = 1; cutoff = 1e-8, kwargs...)
    return retract(ψ, (nretract,); cutoff, kwargs...)
end
