using ITensors
using ITensorQTT
using UnicodePlots

ITensors.disable_warn_order()

include("src/airy_utils.jl")

"""
ns = 7:14
xi, xf = 1.0, 5.0
α, β = 1.0, 1.0
res = airy_compare_to_matrix.(ns; α, β, xi, xf)
airy_err = first.(res)
integral_err = last.(res)
linreg(ns, log2.(airy_err)) # ∫dx |A(x,x')u(x') - b(x)|
# 2-element Vector{Float64}:
#   0.6864681664612592
#  -1.997374659387574
linreg(ns, log2.(integral_err)) # ∫dx |u(x) - ũ(x)|²
# 2-element Vector{Float64}:
#   1.1385702262631532
#  -1.9818952100419311
"""
function airy_compare_to_matrix(n; α, β, xi, xf)
  α, β = (α, β) ./ norm((α, β))
  N = 2^n
  s = siteinds("Qubit", n)

  h = (xf - xi) / N
  # xs = range(; start=xi, stop=xf, length=(N+1))[1:N]
  xs = [xi + j * h for j in 0:(N - 1)]

  u_exact = airy_solution.(xs, α, β)

  Ab_mpo = airy_system(s, xi, xf, α, β)
  A_mpo = mpo_to_mat(Ab_mpo.A)
  b_mpo = mps_to_discrete_function(Ab_mpo.b)
  u_mpo = A_mpo \ b_mpo

  Ab_mat = airy_system_matrix(s, xi, xf, α, β)
  A_mat = Ab_mat.A
  b_mat = Ab_mat.b
  u_mat = A_mat \ b_mat

  display(lineplot(u_exact; label="u_exact"))

  println("From MPO")
  display(A_mpo)
  display(lineplot(u_mpo; label="u_mpo"))

  println("Matrix version")
  display(A_mat)
  display(lineplot(u_mat; label="u_mat"))

  display(lineplot(abs2.(u_mat - u_exact); label="|u_mat - u_exact|²"))

  @show n, N
  @show sum(abs2, u_mat - u_exact) / N
  @show sum(abs2, u_mpo - u_exact) / N
  @show norm(A_mat * u_exact - b_mat)
  @show sum(abs, A_mat * u_exact - b_mat) / N
  @show norm(A_mat * u_mat - b_mat)

  @show diff(diag(A_mpo) / h^2)[1:min(N - 1, 10)]
  @show diff(diag(A_mat) / h^2)[1:min(N - 1, 10)]
  @show (xf - xi) / (N - 1)
  @show (xf - xi) / N

  @assert b_mpo ≈ b_mat
  @assert A_mpo ≈ A_mat
  return sum(abs, A_mat * u_exact - b_mat) / N, sum(abs2, u_mat - u_exact) / N
end
