using ITensors
using ITensorMPS
using ITensorQTT
using JLD2
using Plots
using Random
using UnicodePlots
using FFTW
using MKL

ITensors.disable_warn_order()

include("src/airy_utils.jl")

function projected_linear_system(A, x, b, p, j; ncenter=1)
  n = length(A)
  p = sim(linkinds, p)

  L = ITensor(1.0)
  for i in 1:(j - 1)
    L = L * x[i] * A[i] * p[i]
  end
  R = ITensor(1.0)
  for i in reverse((j + ncenter):n)
    R = R * x[i] * A[i] * p[i]
  end
  Ap = L
  for i in j:(j + ncenter - 1)
    Ap *= A[i]
  end
  Ap *= R

  l = ITensor(1.0)
  for i in 1:(j - 1)
    l = l * b[i] * p[i]
  end
  r = ITensor(1.0)
  for i in reverse((j + ncenter):n)
    r = r * b[i] * p[i]
  end

  bp = l
  for i in j:(j + ncenter - 1)
    bp *= b[i]
  end
  bp *= r

  xc = ITensor(1.0)
  for i in j:(j + ncenter - 1)
    xc *= x[i]
  end

  return Ap, xc, bp
end

function matricize(A::ITensor, left_inds, right_inds)
  return reshape(array(A, left_inds..., right_inds...), dim(left_inds), dim(right_inds))
end

function Base.vec(A::ITensor, inds)
  return vec(array(A, inds))
end

function matricize_linear_system(A, x, b)
  left_inds = uniqueinds(A, x)
  right_inds = commoninds(A, x)
  A_mat = matricize(A, left_inds, right_inds)
  x_vec = vec(x, right_inds)
  b_vec = vec(b, left_inds)
  return A_mat, x_vec, b_vec
end

nxf = 6
n = 10

xi = 1.0
xf = 2^nxf

α, β = 1/√2, 1/√2

filename = "$(ENV["HOME"])/workdir/ITensorQTT.jl/airy_solver_pseudoinverse/results/airy_xf_$(xf)_n_$(n).jld2"
u_init = load(filename, "u")
s = siteinds(u_init)

# Set up Au = b
(; A, b) = airy_system(s, xi, xf, α, β)
# A = A / norm(b)
# b = b / norm(b)
b = b'

r = -(b, contract(A, u_init; cutoff=1e-8); cutoff=1e-8)

display(lineplot(mps_to_discrete_function(u_init)))
@show norm(contract(A, u_init; cutoff=1e-15) - b)
@show maxlinkdim(b)
@show maxlinkdim(u_init)

j = n ÷ 2
orthogonalize!(u_init, j)
orthogonalize!(b, j)
ncenter = 1

println("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
println("Project onto x")
p = copy(u_init')
orthogonalize!(p, j)
@show maxlinkdim(p)
Ap, up_init, bp = projected_linear_system(A, u_init, b, p, j; ncenter)
Ap_mat, up_init_vec, bp_vec = matricize_linear_system(Ap, up_init, bp)
@show norm(Ap_mat * up_init_vec - bp_vec)
up_vec = svd(Ap_mat) \ bp_vec
@show norm(Ap_mat * up_vec - bp_vec)
@show norm(up_vec - up_init_vec)

println("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
println("Project onto b")
p = copy(b)
orthogonalize!(p, j)
@show maxlinkdim(p)
Ap, up_init, bp = projected_linear_system(A, u_init, b, p, j; ncenter)
Ap_mat, up_init_vec, bp_vec = matricize_linear_system(Ap, up_init, bp)
@show norm(Ap_mat * up_init_vec - bp_vec)
up_vec = svd(Ap_mat) \ bp_vec
@show norm(Ap_mat * up_vec - bp_vec)
@show norm(up_vec - up_init_vec)

println("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
println("Project onto b + x")
p = +(b, u_init'; cutoff=1e-8)
orthogonalize!(p, j)
@show maxlinkdim(p)
Ap, up_init, bp = projected_linear_system(A, u_init, b, p, j; ncenter)
Ap_mat, up_init_vec, bp_vec = matricize_linear_system(Ap, up_init, bp)
@show norm(Ap_mat * up_init_vec - bp_vec)
up_vec = svd(Ap_mat) \ bp_vec
@show norm(Ap_mat * up_vec - bp_vec)
@show norm(up_vec - up_init_vec)

println("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
println("Project onto x_trunc")
p = truncate(u_init'; maxdim=(maxlinkdim(u_init)-3))
p = +(p, b; cutoff=1e-15)
p = +(p, r; cutoff=1e-15)
@show maxlinkdim(p)
orthogonalize!(p, j)
Ap, up_init, bp = projected_linear_system(A, u_init, b, p, j; ncenter)
Ap_mat, up_init_vec, bp_vec = matricize_linear_system(Ap, up_init, bp)
@show norm(Ap_mat * up_init_vec - bp_vec)
# up_vec = svd(Ap_mat) \ bp_vec
up_vec = svd(Ap_mat'Ap_mat) \ (Ap_mat'bp_vec)
@show norm(Ap_mat * up_vec - bp_vec)
@show norm(up_vec - up_init_vec) / norm(up_init_vec)

# linsolve_kwargs = (;
#   nsweeps=1,
#   nsite=1,
#   cutoff=1e-15,
#   outputlevel=1,
#   tol=1e-15,
#   krylovdim=30,
#   maxiter=30,
#   ishermitian=false,
# )
# u = b_linsolve(A, b, u_init; linsolve_kwargs...)
# display(lineplot(mps_to_discrete_function(u)))
# @show norm(apply(A, u; cutoff=1e-15) - b)
