function integrate_mps(ψ::MPS)
    s = siteinds(ψ)
    ψ⁺ = MPS([itensor([1 / 2 1 / 2], sⱼ) for sⱼ in s])
    return inner(ψ⁺, ψ)
end

function integrate_mps(A::MPO, ψ::MPS)
    s = siteinds(ψ)
    ψ⁺ = MPS([itensor([1 / 2 1 / 2], sⱼ) for sⱼ in s])
    return inner(ψ⁺', A, ψ)
end
