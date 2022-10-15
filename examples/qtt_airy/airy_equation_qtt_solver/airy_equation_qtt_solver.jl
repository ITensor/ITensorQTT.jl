using ITensors
using ITensorPartialDiffEq
using Random

ITensors.disable_warn_order()

include("../src/linsolve.jl")
include("../src/airy_utils.jl")

# Solve the Airy equation:
#
# u''(x) + q(x) u(x) = 0
#
# where:
#
# q(x) = -x
#
# with the boundary values:
#
# u(xⁱ) = uⁱ
# u(xᶠ) = uᶠ
function airy_qtt_solver(;
  n=nothing,
  s=siteinds("Qubit", n),
  xⁱ=1.0,
  xᶠ,
  α,
  β,
  seed=1234,
  init=nothing,
  linsolve_kwargs=(; nsweeps=10, cutoff=1e-15, outputlevel=1, tol=1e-15, krylovdim=30, maxiter=30, ishermitian=true),
)
  # Normalize the coefficients
  α, β = (α, β) ./ norm((α, β))

  n = length(s)

  @show n, 2^n, n * log10(2)
  @show xⁱ, xᶠ
  @show linsolve_kwargs

  time = @elapsed begin
    # Set up Au = b
    (; A, b) = airy_system(s, xⁱ, xᶠ, α, β)

    # Starting guess for solution
    if isnothing(init)
      Random.seed!(seed)
      u = randomMPS(s; linkdims=10)
    else
      init = replace_siteinds(init, s[1:length(init)])
      u = prolongate(init, s[length(init) + 1:end])
    end

    # Solve Au = b
    u = linsolve(A, b, u; nsite=2, linsolve_kwargs...)
    u = linsolve(A, b, u; nsite=1, linsolve_kwargs...)
  end

  @show linsolve_error(A, u, b)

  return (; u, xⁱ, xᶠ, α, β, seed, linsolve_kwargs, time)
end
