function fourier_interpolation(ψ::MPS, s_interp::Vector{<:Index}; cutoff = 1e-15)
    s = siteinds(ψ)
    ℱψ = apply_dft_mpo(ψ; cutoff)
    ℱψ0 = project_bits(ℱψ, [0])
    ℱψ1 = project_bits(ℱψ, [1])
    ℱψ0s = [[onehot.(s_interp .=> "0"); onehot(s[1] => "0")]; ℱψ0]
    ℱψ1s = [[onehot.(s_interp .=> "1"); onehot(s[1] => "1")]; ℱψ1]
    ℱψ_interp = +(ℱψ0s, ℱψ1s; cutoff = 1e-15)
    return apply_idft_mpo(ℱψ_interp; cutoff) * 2^(length(s_interp) / 2)
end

"""
Fourier interpolation of MPS/QTT from https://arxiv.org/abs/1909.06619
Interpolate an MPS/QTT ψ of length `n` to an MPS/QTT of length
`n + k`.
"""
function fourier_interpolation(ψ::MPS, k::Int = 1; kwargs...)
    return fourier_interpolation(ψ, siteinds("Qubit", k); kwargs...)
end

function repeat_function(ψ::MPS, sₖ::Vector{<:Index}; cutoff = 1e-15)
    s = siteinds(ψ)
    n = length(s)
    k = length(sₖ)
    ℱψ = apply_dft_mpo(ψ; cutoff)
    # Insert qubits in position 1:k
    sₙ₊ₖ = [sₖ; s]
    ℱψₖ = iszero(k) ? MPS(ITensor[]) : MPS(sₖ, "0")
    ℱψₙ₊ₖ = [ℱψ; ℱψₖ]
    return apply_idft_mpo(ℱψₙ₊ₖ; cutoff) * 2^(length(sₖ) / 2)
end

"""
Repeat a function `2ᵏ` times.
"""
function repeat_function(ψ::MPS, k::Int; cutoff = 1e-15)
    return repeat_function(ψ, siteinds("Qubit", k); cutoff)
end
