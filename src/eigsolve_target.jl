# Find the eigenvalue of `A` closest to `λ`
function eigsolve_target(A, λ::Number, x₀; linsolve_kwargs=(;), eigsolve_kwargs=(;))
  function f(x)
    x′, info = linsolve(A, x, x, -λ; linsolve_kwargs...)
    return x′
  end
  vals, vecs, info = eigsolve(f, x₀; eigsolve_kwargs...)
  return vecs[1]
end
