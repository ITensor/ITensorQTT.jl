using ITensors
using ITensorMPS
using ITensorQTT

k = 2π * 2^18
n = 21

s = siteinds("Qubit", n)
h = 1 / 2^n
A = laplacian_mpo(s) / h^2
u_init = randomMPS(s; linkdims=10)
u = dmrg_target(A, u_init; target_eigenvalue=-k^2, cutoff=1e-15, nsweeps=10, outputlevel=1)
inner(u', A, u) / inner(u, u), -k^2
