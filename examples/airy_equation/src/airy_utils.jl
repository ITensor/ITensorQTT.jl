using ITensors
using ITensorMPS
using ITensorQTT
using SpecialFunctions
using LinearAlgebra
using JLD2

airy_solution(x, α=1.0, β=0.0) = α * airyai(-x) + β * airybi(-x)

function airy_mpo(s, xi, xf)
  s̃ = sim(s)
  n = length(s)
  A₁ = -laplacian_mpo(s, 1.0)

  # N = 2^n
  # h = (xf - xi) / (N - 1)
  # f(x) = -((xf + h - xi) * x + xi)

  h = (xf - xi) / 2^n
  f(x) = -((xf - xi) * x + xi)

  q_mps = function_to_mps(
    f, s̃, xi, xf; cutoff=1e-8, alg="polynomial", degree=1, length=1000
  )
  A₂ = h^2 * MPO([q_mps[j] * δ(s̃[j], s[j], s[j]') for j in 1:n])
  return convert(MPO, +(A₁, A₂; alg="directsum"))
end

function airy_system(s, xi, xf, α, β)
  # Normalize the coefficients
  α, β = (α, β) ./ norm((α, β))
  A = airy_mpo(s, xi, xf)
  ui, uf = airy_solution.((xi, xf), α, β)
  b = boundary_value_mps(s, ui, uf)
  return (; A, b)
end

# Airy equation matrix in the region `[xi, xf) = [xi, xf - h]`.
# [xi + j * h for j in 0:(N - 1)]
# h = (xf - xi) / N
function airy_matrix(s, xi, xf)
  n = length(s)
  N = 2^n
  h = (xf - xi) / N
  A₁ = SymTridiagonal(fill(2.0, N), fill(-1.0, N - 1))
  A₂ = h^2 * Diagonal([-(xi + j * h) for j in 0:(N - 1)])
  return A₁ + A₂
end

function airy_system_matrix(s, xi, xf, α, β)
  # Normalize the coefficients
  α, β = (α, β) ./ norm((α, β))
  A = airy_matrix(s, xi, xf)
  ui, uf = airy_solution.((xi, xf), α, β)
  b = boundary_value_vector(s, ui, uf)
  return (; A, b)
end

# Saving and loading data
airy_solver_filename(xf, n) = "airy_xf_$(xf)_n_$(n).jld2"
airy_solver_filename(; dirname, xf, n) = joinpath(dirname, airy_solver_filename(xf, n))

function load_airy_results(; dirname, xf, n)
  filename = airy_solver_filename(; dirname, xf, n)
  println("Loading results from $(filename)")
  res = load(filename)
  return (; (Symbol.(keys(res)) .=> values(res))...)
end

function airy_qtt(s::Vector{<:Index}, xi, xf; α=1.0, β=0.0, cutoff=1e-15)
  α, β = (α, β) ./ norm((α, β))
  return function_to_mps(x -> airy_solution(x, α, β), s, xi, xf; cutoff)
end

function airy_qtt_compression(n::Int, xi, xf; α=1.0, β=0.0, cutoff=1e-15)
  α, β = (α, β) ./ norm((α, β))

  # Exact Airy solution from SpecialFunctions.jl
  xrange = qtt_xrange(n, xi, xf)
  u_vec_exact = airy_solution.(xrange, α, β)

  s = siteinds("Qubit", n)
  u = vec_to_mps(u_vec_exact, s; cutoff)

  return (; u, u_vec_exact, α, β, cutoff, xi, xf)
end
