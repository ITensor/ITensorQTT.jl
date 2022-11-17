using ITensors
using ITensorQTT
using Polynomials
using FiniteDifferences
using UnicodePlots
using ToeplitzMatrices
using LinearAlgebra

ITensors.disable_warn_order()

n = 10
s = siteinds("Qubit", n)

# https://en.wikipedia.org/wiki/Finite_difference_coefficient
# https://discourse.julialang.org/t/generating-finite-difference-stencils/85876
# https://juliadiff.org/FiniteDifferences.jl/stable/
# https://people.maths.ox.ac.uk/trefethen/7all.pdf
order = 1
fd = central_fdm(2order + 1, 2)
fd_matrix = SymmetricToeplitz([-fd.coefs[(order + 1):end]; zeros(2^n - order - 1)])
D, U = eigen(fd_matrix)
display(lineplot(D; title="Eigenvalues of finite difference matrix"))
display(lineplot(U[:, 1]; title="Eigenvector 1 of finite difference matrix"))
display(lineplot(U[:, 2]; title="Eigenvector 2 of finite difference matrix"))

ℱ = ITensorQTT.dft_matrix(n)

xs = range(; start=0, stop=1, length=(2^n+1))[1:2^n]
ds = (-cos.(π * xs) .+ 1) * maximum(D) / 2
display(lineplot(ds; title="Exact eigenvalues"))
display(lineplot(abs.(ds .- D); title="Difference from exact eigenvalues"))

# Spectral representation of the differential operator
# https://arxiv.org/abs/1909.06619
# ∂ₓ → ik
# ∂ₓₓ → -k²
# This has periodic boundary conditions, how to change
# to open boundary conditions?
# D = spectral_differential(Polynomial([0, 0, 1]), s)
# @show maxlinkdim(D[1])
# A = mpo_to_mat(D[1])
# display(lineplot(abs.(A[:, 1])))
# display(lineplot(abs.(A[:, 2^(n-1)])))
# display(lineplot(eigen(Hermitian(A)).values))

