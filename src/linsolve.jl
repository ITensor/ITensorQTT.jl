import Base: copy
using ITensorMPS: ITensorMPS

function LinearAlgebra.pinv(T::ITensor, left_inds...; kwargs...)
  U, S, V = svd(T, left_inds...; kwargs...)
  S⁻¹ = copy(S)
  for j in 1:ITensors.diaglength(S)
    S⁻¹[j, j] = inv(S[j, j])
  end
  return dag(U) * dag(S⁻¹) * dag(V)
end

function proj(P::ITensorMPS.ProjMPS)
  ϕ = prime(linkinds, P.M)
  p = ITensor(1.0)
  !isnothing(lproj(P)) && (p *= lproj(P))
  for j in (P.lpos + 1):(P.rpos - 1)
    p *= dag(ϕ[j])
  end
  !isnothing(rproj(P)) && (p *= rproj(P))
  return dag(p)
end

# Get `l * b * r`
function get_b(P::ITensorMPS.ProjMPS)
  # ϕ = prime(linkinds, P.M)
  ϕ = P.M

  b = ITensor(1.0)
  for j in (P.lpos + 1):(P.rpos - 1)
    b *= dag(ϕ[j])
  end

  l = ITensor(1.0)
  !isnothing(lproj(P)) && (l *= lproj(P))

  r = ITensor(1.0)
  !isnothing(rproj(P)) && (r *= rproj(P))

  return (; b=dag(b), l=dag(noprime(l)), r=dag(noprime(r)))
end

# Compute a solution x to the linear system:
#
# (a₀ + a₁ * A)*x = b
#
# using starting guess x₀.
#
# Solve using a pseudoinverse of `A`, helps when `A` is singular.
# 
# WARNING: currently scales as `chi^4` since it builds the full
# matrix for the projected MPO A. Need to investigate Krylov-based
# methods.
function linsolve_pseudoinverse(
  A::MPO,
  b::MPS,
  x₀::MPS,
  a₀::Number=0,
  a₁::Number=1;
  reverse_step=false,
  tol=1e-5,
  krylovdim=20,
  maxiter=10,
  ishermitian=false,
  kwargs...,
)
  function linsolve_solver(PH, t, x₀; kwargs...)
    A = PH.PH
    (; b, l, r) = get_b(only(PH.pm))

    T = lproj(A)
    for h in A.H[ITensors.site_range(A)]
      T *= h
    end
    T *= rproj(A)

    blr = (b * l * r)'
    b_inds = inds(blr)
    x_inds = inds(x₀)
    Cb = combiner(b_inds)
    cb = combinedind(Cb)
    Cx = combiner(x_inds)
    cx = combinedind(Cx)
    T_mat = matrix(T * Cb * Cx, cb, cx)
    b_vec = vector(blr * Cb)

    # F = svd(T_mat)
    F = qr!(T_mat)
    x_vec = F \ b_vec
    x = itensor(x_vec, cx) * dag(Cx)

    return x, nothing
  end
  t = Inf
  PH = ProjMPO_MPS(A, [b])
  return tdvp(linsolve_solver, PH, t, x₀; reverse_step, kwargs...)
end

# Try some preconditioning, not working right now.
function linsolve_precondition(
  A::MPO,
  b::MPS,
  x₀::MPS,
  a₀::Number=0,
  a₁::Number=1;
  reverse_step=false,
  tol=1e-5,
  krylovdim=20,
  maxiter=10,
  ishermitian=false,
  kwargs...,
)
  function linsolve_solver(PH, t, x₀; kwargs...)
    A = PH.PH
    (; b, l, r) = get_b(only(PH.pm))

    T = lproj(A)
    for h in A.H[ITensors.site_range(A)]
      T *= h
    end
    T *= rproj(A)

    if order(l) > 0
      l⁻¹ = pinv(l, ind(l, 1); cutoff=1e-15)
    else
      l⁻¹ = ITensor(inv(l[]))
    end

    if order(r) > 0
      r⁻¹ = pinv(r, ind(r, 1); cutoff=1e-15)
    else
      r⁻¹ = ITensor(inv(r[]))
    end

    x̃₀ = x₀
    # x̃₀ *= l⁻¹
    # x̃₀ *= r⁻¹

    T̃ = T
    T̃ *= r⁻¹'
    T̃ *= l⁻¹'
    # T̃ *= r
    # T̃ *= l

    b = b'
    b_inds = inds(b)
    x_inds = inds(x̃₀)

    # @show b_inds
    # @show x_inds
    # @show inds(T̃)

    Cb = combiner(b_inds)
    cb = combinedind(Cb)
    Cx = combiner(x_inds)
    cx = combinedind(Cx)
    T_mat = matrix(T̃ * Cb * Cx, cb, cx)
    b_vec = vector(b * Cb)
    x_vec = svd(T_mat) \ b_vec
    x̃ = itensor(x_vec, cx) * dag(Cx)

    # x̃, info = linsolve(T̃, b, x̃₀, a₀, a₁; ishermitian, tol, krylovdim, maxiter)

    x = x̃
    # x *= l
    # x *= r

    return x, nothing
  end
  t = Inf
  PH = ProjMPO_MPS(A, [b])
  return tdvp(linsolve_solver, PH, t, x₀; reverse_step, kwargs...)
end

mutable struct ProjMPOLinsolve <: ITensorMPS.AbstractProjMPO
  lpos::Int
  rpos::Int
  nsite::Int
  x::MPS
  b::MPS
  H::MPO
  LR::Vector{ITensor}
end

function ProjMPOLinsolve(x0::MPS, b::MPS, A::MPO)
  return ProjMPOLinsolve(0, length(A) + 1, 2, x0, b, A, Vector{ITensor}(undef, length(A)))
end

function copy(P::ProjMPOLinsolve)
  return ProjMPOLinsolve(
    P.lpos, P.rpos, P.nsite, copy(P.x), copy(P.b), copy(P.H), copy(P.LR)
  )
end

function set_nsite!(P::ProjMPOLinsolve, nsite)
  P.nsite = nsite
  return P
end

function makeL!(P::ProjMPOLinsolve, x::MPS, k::Int)
  # Save the last `L` that is made to help with caching
  # for DiskProjMPO
  ll = P.lpos
  if ll ≥ k
    # Special case when nothing has to be done.
    # Still need to change the position if lproj is
    # being moved backward.
    P.lpos = k
    return nothing
  end
  # Make sure ll is at least 0 for the generic logic below
  ll = max(ll, 0)
  L = lproj(P)
  while ll < k
    L = L * x[ll + 1] * P.H[ll + 1] * dag(P.b[ll + 1])
    P.LR[ll + 1] = L
    ll += 1
  end
  # Needed when moving lproj backward.
  P.lpos = k
  return P
end

function makeR!(P::ProjMPOLinsolve, x::MPS, k::Int)
  # Save the last `R` that is made to help with caching
  # for DiskProjMPO
  rl = P.rpos
  if rl ≤ k
    # Special case when nothing has to be done.
    # Still need to change the position if rproj is
    # being moved backward.
    P.rpos = k
    return nothing
  end
  N = length(P.H)
  # Make sure rl is no bigger than `N + 1` for the generic logic below
  rl = min(rl, N + 1)
  R = rproj(P)
  while rl > k
    R = R * x[rl - 1] * P.H[rl - 1] * dag(P.b[rl - 1])
    P.LR[rl - 1] = R
    rl -= 1
  end
  P.rpos = k
  return P
end

function position!(P::ProjMPOLinsolve, x::MPS, pos::Int)
  ITensors.orthogonalize!(P.b, pos)
  ITensorMPS.makeL!(P, x, pos - 1)
  ITensorMPS.makeR!(P, x, pos + nsite(P))
  return P
end

function bvec(P::ProjMPOLinsolve)
  bv = ITensor(1.0)
  for j in (P.lpos + 1):(P.rpos - 1)
    bv *= P.b[j]
  end
  return bv
end

function b_linsolve(
  A::MPO, b::MPS, x₀::MPS, a₀::Number=0, a₁::Number=1; reverse_step=false, kwargs...
)
  function b_linsolve_solver(P::ProjMPOLinsolve, t, x₀; kws...)
    solver_kwargs = (;
      ishermitian=get(kws, :ishermitian, false),
      tol=get(kws, :solver_tol, 1E-14),
      krylovdim=get(kws, :solver_krylovdim, 30),
      maxiter=get(kws, :solver_maxiter, 100),
      verbosity=get(kws, :solver_verbosity, 0),
    )
    b = noprime(bvec(P))
    x, info = KrylovKit.linsolve(P, b, x₀, a₀, a₁; solver_kwargs...)
    return x, nothing
  end

  function b_linsolve_exact_solver(P::ProjMPOLinsolve, t, x₀; kws...)
    bT = bvec(P)
    T = ITensor(1.0)
    let
      itensor_map = Union{ITensor,ITensors.OneITensor}[lproj(P)]
      append!(itensor_map, P.H[ITensors.site_range(P)])
      push!(itensor_map, rproj(P))
      for it in itensor_map
        T *= it
      end
    end
    rowi = commoninds(T, bT)
    coli = uniqueinds(T, bT)
    Dr = prod(dim.(rowi))
    Dc = prod(dim.(coli))
    b = reshape(array(bT, rowi...), Dr)
    A = reshape(array(T, rowi..., coli...), Dr, Dc)
    F = svd(A)
    x = F \ b
    xT = ITensor(x, coli...)
    return xT, nothing
  end

  t = Inf
  b = prime(b, "Site") # TODO: generalize
  P = ProjMPOLinsolve(x₀, b, A)
  return tdvp(b_linsolve_exact_solver, P, t, x₀; reverse_step, kwargs...)
  #return tdvp(b_linsolve_solver, P, t, x₀; reverse_step, kwargs...)
end
