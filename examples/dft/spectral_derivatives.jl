using ITensors
using ITensorQTT
using Polynomials
using FiniteDifferences

ITensors.disable_warn_order()

n = 8
s = siteinds("Qubit", n)

# Spectral representation of the differential operator
# https://arxiv.org/abs/1909.06619
# ∂ₓ → ik
# ∂ₓₓ → -k²
# This has periodic boundary conditions, how to change
# to open boundary conditions?
D = spectral_differential(Polynomial([0, 0, 1]), s)
@show maxlinkdim(D[1])
A = mpo_to_mat(D[1])
display(lineplot(abs.(A[:, 1])))
display(lineplot(abs.(A[:, 2^(n-1)])))

# https://en.wikipedia.org/wiki/Finite_difference_coefficient
# https://discourse.julialang.org/t/generating-finite-difference-stencils/85876
# https://juliadiff.org/FiniteDifferences.jl/stable/
# https://people.maths.ox.ac.uk/trefethen/7all.pdf
@show central_fdm(5, 2)

