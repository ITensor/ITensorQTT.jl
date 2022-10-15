function integrate_mps(ψ::MPS)
  s = siteinds(ψ)
  return inner(MPS(s, "+") ./ √(2), ψ)
end
