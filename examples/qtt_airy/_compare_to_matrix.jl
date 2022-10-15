using ITensors
using ITensorPartialDiffEq
using UnicodePlots

using ITensorPartialDiffEq: mpo_to_mat

include("airy_utils.jl")

n = 3
N = 2^n
s = siteinds("Qubit", n)

xⁱ = 1.0
xᶠ = 5.0
α = 1.0
β = 1.0

h = (xᶠ - xⁱ) / (N - 1)

Ab_mpo = airy_system(s, xⁱ, xᶠ, α, β)
A_mpo = mpo_to_mat(Ab_mpo.A)
b_mpo = mps_to_discrete_function(Ab_mpo.b)

Ab_mat = airy_system_matrix(s, xⁱ, xᶠ, α, β)
A_mat = Ab_mat.A
b_mat = Ab_mat.b

println("From MPO")
display(A_mpo)
display(lineplot(A_mpo \ b_mpo))

println("Matrix version")
display(A_mat)
display(lineplot(A_mat \ b_mat))

@show diff(diag(A_mpo) / h^2)
@show diff(diag(A_mat) / h^2)
@show (xᶠ - xⁱ) / (N - 1)
@show (xᶠ - xⁱ) / N

@assert A_mpo ≈ A_mat
@assert b_mpo ≈ b_mat
