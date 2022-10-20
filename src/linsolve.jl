using ITensors: ProjMPS

function proj(P::ProjMPS)
  ϕ = prime(linkinds, P.M)
  p = ITensor(1.0)
  !isnothing(lproj(P)) && (p *= lproj(P))
  for j in (P.lpos + 1):(P.rpos - 1)
    p *= dag(ϕ[j])
  end
  !isnothing(rproj(P)) && (p *= rproj(P))
  return dag(p)
end

# Compute a solution x to the linear system:
#
# (a₀ + a₁ * A)*x = b
#
# using starting guess x₀.
function KrylovKit.linsolve(A::MPO, b::MPS, x₀::MPS, a₀::Number=0, a₁::Number=1; reverse_step=false, tol=1e-5, krylovdim=20, maxiter=10, ishermitian=false, kwargs...)
  function linsolve_solver(PH, t, x₀; kwargs...)
    A = PH.PH
    b = proj(only(PH.pm))
    x, info = linsolve(A, b, x₀, a₀, a₁; ishermitian, tol, krylovdim, maxiter)
    return x, nothing
  end
  t = Inf
  PH = ProjMPO_MPS(A, [b])
  return tdvp(linsolve_solver, PH, t, x₀; reverse_step, kwargs...)
end
