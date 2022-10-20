#
# Making differential operators as MPOs
# https://arxiv.org/abs/1802.07259
# https://en.wikipedia.org/wiki/Finite_difference_method
# TODO: higher order discetizations
# https://en.wikipedia.org/wiki/Finite_difference_coefficient
#

function laplacian_matrix(xlength, xstep=1.0)
  a = fill(-2.0, xlength)
  b = fill(1.0, xlength - 1)
  A = SymTridiagonal(a, b)
  return A / (xstep ^ 2)
end

# https://arxiv.org/abs/1802.07259
function laplacian_tensor()
  T = zeros(3, 3, 2, 2)
  T[1, 1, 1, 1] = 1.0
  T[1, 1, 2, 2] = 1.0
  T[2, 2, 2, 1] = 1.0
  T[1, 2, 1, 2] = 1.0
  T[1, 3, 2, 1] = 1.0
  T[3, 3, 1, 2] = 1.0
  return T
end

# https://arxiv.org/abs/1802.07259
function laplacian_mpo(s::Vector{<:Index}, xstep=1.0)
  L = length(s)
  l = [Index(3, "l=$(j)↔$(j+1)") for j in 0:L]
  T⃗ = [itensor(laplacian_tensor(), l[j], l[j + 1], s[j], s[j]') for j in 1:L]
  # Left boundary condition
  T⁰ = onehot(l[1] => 1)
  T⃗[1] *= T⁰
  # Right boundary condition
  Tᴸ⁺¹ = ITensor(l[L + 1])
  Tᴸ⁺¹[1] = -2.0
  Tᴸ⁺¹[2] = 1.0
  Tᴸ⁺¹[3] = 1.0
  T⃗[L] *= Tᴸ⁺¹
  return MPO(T⃗) ./ (xstep ^ (2 / L))
end

function laplacian_mpo(s::Tuple{Vector{<:Index},Vector{<:Index}}, xstep::Tuple{Number,Number}=(1.0, 1.0))
  Δ₁ = interleave(laplacian_mpo(s[1], xstep[1]), MPO(s[2], "I"))
  Δ₂ = interleave(MPO(s[1], "I"), laplacian_mpo(s[2], xstep[2]))
  return +(insert_missing_links(Δ₁), insert_missing_links(Δ₂); alg="directsum")
end

#
# Ŝ⁺ ladder tensor
# Ŝ⁺|n⟩ = |n+1⟩
#
#     s'
#     |
# l---T---r
#     |
#     s
#
# T_{l,r,s,s′} = {1 if s′ = s ⊻ r, l = s ∧ r, else 0}
#
# https://arxiv.org/abs/1909.06619, Eq. (59-61) and (66)
function s_plus_tensor()
  T = zeros(2, 2, 2, 2)
  for l in 0:1, r in 0:1, s in 0:1, s′ in 0:1
    if (s′ == s ⊻ r) && (l == s & r)
      T[l + 1, r + 1, s + 1, s′ + 1] = one(eltype(T))
    end
  end
  return T
end

function s_minus_tensor()
  return permutedims(s_plus_tensor(), (1, 2, 4, 3))
end

function s_plus_itensor(l, r, s, s′)
  return itensor(s_plus_tensor(), (l, r, s, s′))
end

function s_minus_itensor(l, r, s, s′)
  return itensor(s_minus_tensor(), (l, r, s, s′))
end

function s_generic_mpo(f, s)
  n = length(s)
  l = [Index(2, "l=$n↔$(n + 1)") for n in 0:(n + 1)]
  M = MPO([f(l[j], l[j + 1], s[j], s[j]') for j in 1:n])
  M[1] *= onehot(l[1] => 1)
  M[n] *= onehot(l[n + 1] => 2)
  return M
end

s_plus_mpo(s) = s_generic_mpo(s_plus_itensor, s)
s_minus_mpo(s) = s_generic_mpo(s_minus_itensor, s)

# Laplacian MPO representation from:
#
# Δ = ∇² = (Ŝ⁺ + Ŝ⁻ - 2I)
function laplacian_mpo_approx(s, xstep=1.0; cutoff=1e-15)
  A = +(s_plus_mpo(s), s_minus_mpo(s), -2 * MPO(s, "I"); cutoff)
  return A ./ (xstep ^ (2 / L))
end
