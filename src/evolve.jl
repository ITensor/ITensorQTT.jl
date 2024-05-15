function evolve(A, u, trange)
  tstep = trange[2] - trange[1]
  u⃗ = Vector{typeof(u)}(undef, length(trange))
  u⃗[1] = u
  for j in 2:length(trange)
    u += A(u) * tstep
    u⃗[j] = u
  end
  return u⃗
end
