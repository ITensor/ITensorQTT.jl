#
# Function to MPS conversion
#

function function_to_mps(f::Function, s::Vector{<:Index}, xstart, xstop; alg="factorize", kwargs...)
  return function_to_mps(Algorithm(alg), f, s, xstart, xstop; kwargs...)
end

# TODO: generalize to n-dimensional functions
function function_to_mps(f::Tuple{Vararg{Function}}, s::Tuple{Vararg{Vector{<:Index}}}, xstart::Tuple{Vararg{Number}}, xstop::Tuple{Vararg{Number}}; kwargs...)
  @assert allequal(length.(s)) #all(==(length(first(s))), length.(s))
  ψ = function_to_mps.(f, s, xstart, xstop; kwargs...)
  return interleave(ψ...)
end

function function_to_mps(::Algorithm"factorize", f::Function, s::Vector{<:Index}, xstart, xstop; cutoff=1e-15, kwargs...)
  x = range(; start=xstart, stop=xstop, length=(2 ^ length(s) + 1))[1:end-1]
  return vec_to_mps(f.(x), s; cutoff, kwargs...)
end

# https://arxiv.org/abs/1802.07259
function polynomial_tensor(j, κ)
  Q = zeros(κ + 1, κ + 1, 2)
  for α in 1:(κ + 1)
    Q[α, α, 1] = 1.0
    Q[α, α, 2] = 1.0
  end
  for β in 2:(κ + 1)
    for α in 1:(β - 1)
      Q[α, β, 2] = binomial(β - 1, α - 1) * 2.0 ^ (-(β - α) * j)
    end
  end
  return Q
end

# https://arxiv.org/abs/1802.07259
# Approximate a function by a polynomial and then turn into an MPS
function function_to_mps(::Algorithm"polynomial", f::Function, s::Vector{<:Index}, xstart, xstop; cutoff=1e-15, degree, length=2^(length(s)))
  # xrange = range(; start=xstart, stop=xstop, length=(length + 1))[1:end-1]
  xrange = range(; start=xstart, stop=xstop, length)
  # ∑{l=0…κ} cₗ xˡ, polynomial of degree κ
  c = coeffs(fit(xrange, f.(xrange), degree))
  κ = degree
  # MPS of bond dimension `κ + 1`
  L = Base.length(s)
  l = [Index(κ + 1, "l=$(j)↔$(j+1)") for j in 0:L]
  Q⃗ = [itensor(polynomial_tensor(j, κ), l[j], l[j + 1], s[j]) for j in 1:L]
  # Left boundary tensor
  Q⁰ = onehot(l[1] => 1)
  Q⃗[1] *= Q⁰
  # Right boundary tensor
  Qᴸ⁺¹ = itensor(c, l[L + 1])
  Q⃗[L] *= Qᴸ⁺¹
  return ITensors.truncate!(MPS(Q⃗); cutoff)
end

# Approximate a function in the range `[xstart, xstop)`
# as an MPS with `L` site indices.
# The function is discretized on a grid with `N = 2^L` points.
# Using recursive/multiscale method from Miles' tensor meeting notes
function function_to_mps(::Algorithm"recursive", f::Function, s::Vector{<:Index}, xstart, xstop; cutoff=1e-15)
  L = length(s)
  N = 2 ^ L
  h = 1 / N
  x = range(; start=xstart, stop=xstop, step=h)[1:end-1]
  @assert length(x) == N
  # Start by making `L - 2` MPS of length `2`, the full basis
  # on sites `[s[L-1], s[L]]`.
  ranges = collect(Iterators.partition(1:N, 4))
  l = 2 # length of the MPS
  ψ = Vector{MPS}(undef, 2 ^ (L - l))
  for j in eachindex(ranges)
    range = ranges[j]
    Aⱼ = [f(x[range[1]]) f(x[range[2]]); f(x[range[3]]) f(x[range[4]])]
    ψ[j] = MPS(Aⱼ, [s[L - 1], s[L]]; cutoff)
  end
  for l in (l + 1):L
    ψ̃ = Vector{MPS}(undef, 2 ^ (L - l))
    for j in 1:(2 ^ (L - l))
      ψ̃[j] = +([onehot(s[L - l + 1] => 1); ψ[2j - 1]], [onehot(s[L - l + 1] => 2); ψ[2j]]; cutoff)
    end
    ψ = ψ̃
  end
  return only(ψ)
end

#
# MPS to function conversion
#

function mps_to_discrete_function(ψ::MPS; kwargs...)
  return mps_to_discrete_function(Val(1), ψ; kwargs...)
end

function mps_to_discrete_function(::Val{ndims}, ψ::MPS) where {ndims}
  n = length(ψ)
  s = siteinds(ψ)
  s⃗ = ntuple(j -> s[j:ndims:n], Val(ndims))
  return reshape(Array(contract(ψ), vcat(reverse.(s⃗)...)), dim.(s⃗))
end

function mps_to_discrete_function(ψ::MPS, nbits::Integer; kwargs...)
  return mps_to_discrete_function(Val(1), ψ, nbits; kwargs...)
end

function mps_to_discrete_function(ψ::MPS, nbits::Tuple{Vararg{Integer}}; kwargs...)
  @assert allequal(nbits)
  ndims = length(nbits)
  nbits_original = ntuple(Returns(length(ψ) ÷ ndims), Val(ndims))
  nbits_retract = nbits_original .- nbits
  ψ = retract(ψ, nbits_retract; kwargs...)
  return mps_to_discrete_function(Val(ndims), ψ)
end

function mps_to_discrete_function(::Val{ndims}, ψ::MPS, nbits::Integer; kwargs...) where {ndims}
  return mps_to_discrete_function(ψ, ntuple(Returns(nbits), Val(ndims)))
end

#
# Vector/MPS conversion
#

function vec_to_tensor(v)
  n = Int(log2(length(v)))
  return permutedims(reshape(v, fill(2, n)...), n:-1:1)
end

function vec_to_mps(v, s; kwargs...)
  return MPS(vec_to_tensor(v), s; kwargs...)
end

#
# Matrix/MPO conversion
#

function mpo_to_mat(m)
  s = only.(siteinds(m; plev=0))
  return reshape((Array(contract(m), reverse(s'), reverse(s))), dim(s), dim(s))
end

function mat_to_mpo(m, s; kwargs...)
  n = length(s)
  t = reshape(Matrix(m), fill(2, 2n)...)
  t = permutedims(t, [n:-1:1; 2n:-1:(n + 1)])
  t = permutedims(t, [1:n (n + 1):2n]'[:])
  s_mpo = [[s[n], s[n]'] for n in 1:length(s)]
  return MPO(t, s_mpo; kwargs...)
end
