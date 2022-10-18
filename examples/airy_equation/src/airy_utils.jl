using ITensors
using SpecialFunctions
using LinearAlgebra
using JLD2

number_of_zeros(v) = count(j -> sign(v[j]) ≠ sign(v[j+1]), 1:(length(v)-1))

linreg(x, y) = [fill!(similar(x), 1);; x] \ y

function linsolve_error(A, x, b)
  return √(abs(inner(A, x, A, x) + inner(b, b) - 2 * real(inner(b', A, x)))) / norm(b)
end

function boundary_value_mps(s, xⁱ, xᶠ)
  n = length(s)
  l = [Index(2; tags="Link,l=$(j)↔$(j+1)") for j in 1:(n - 1)]
  A = MPS(n)
  A[1] = itensor([1.0 0.0; 0.0 1.0], s[1], l[1])
  aⱼ = zeros(2, 2, 2)
  aⱼ[1, 1, 1] = 1.0
  aⱼ[2, 2, 2] = 1.0
  for j in 2:(n - 1)
    A[j] = ITensor(aⱼ, l[j - 1], s[j], l[j])
  end
  A[end] = itensor([xⁱ 0.0; 0.0 xᶠ], l[n - 1], s[n])
  return A
end

q(x) = -x

airy_solution(x, α=1.0, β=0.0) = α * airyai(-x) + β * airybi(-x)

function airy_mpo(s, xⁱ, xᶠ)
  s̃ = sim(s)
  n = length(s)
  A₁ = -laplacian_mpo(s, 1.0)

  # N = 2^n
  # h = (xᶠ - xⁱ) / (N - 1)
  # f(x) = q((xᶠ + h - xⁱ) * x + xⁱ)

  h = (xᶠ - xⁱ) / 2^n
  f(x) = q((xᶠ - xⁱ) * x + xⁱ)

  q_mps = function_to_mps(f, s̃, xⁱ, xᶠ; cutoff=1e-8, alg="polynomial", degree=1, length=1000)
  A₂ = h^2 * MPO([q_mps[j] * δ(s̃[j], s[j], s[j]') for j in 1:n])
  return convert(MPO, +(A₁, A₂; alg="directsum"))
end

function airy_system(s, xⁱ, xᶠ, α, β)
  # Normalize the coefficients
  α, β = (α, β) ./ norm((α, β))
  A = airy_mpo(s, xⁱ, xᶠ)
  uⁱ, uᶠ = airy_solution.((xⁱ, xᶠ), α, β)
  b = boundary_value_mps(s, uⁱ, uᶠ)
  return (; A, b)
end

# Airy equation matrix in the region `[xi, xf) = [xi, xf - h]`.
# [xi + j * h for j in 0:(N - 1)]
# h = (xf - xi) / N
function airy_matrix(s, xⁱ, xᶠ)
  n = length(s)
  N = 2^n
  h = (xᶠ - xⁱ) / N
  A₁ = SymTridiagonal(fill(2.0, N), fill(-1.0, N - 1))
  A₂ = h^2 * Diagonal([-(xⁱ + j * h) for j in 0:(N - 1)])
  return A₁ + A₂
end

function boundary_value_vector(s, uⁱ, uᶠ)
  n = length(s)
  N = 2^n
  b = zeros(N)
  b[1] = uⁱ
  b[N] = uᶠ
  return b
end

function airy_system_matrix(s, xⁱ, xᶠ, α, β)
  # Normalize the coefficients
  α, β = (α, β) ./ norm((α, β))
  A = airy_matrix(s, xⁱ, xᶠ)
  uⁱ, uᶠ = airy_solution.((xⁱ, xᶠ), α, β)
  b = boundary_value_vector(s, uⁱ, uᶠ)
  return (; A, b)
end

# Saving and loading data
airy_filename(; dirname, xᶠ, n) = "$(dirname)/airy_xf_$(xᶠ)_n_$(n).jld2"

function load_airy_results(; dirname, xᶠ, n)
  filename = airy_filename(; dirname, xᶠ, n)
  println("Loading results from $(filename)")
  res = load(filename)
  return (; (Symbol.(keys(res)) .=> values(res))...)
end
