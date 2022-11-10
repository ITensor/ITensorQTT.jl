"""
    qft(n::Int; inverse::Bool = false)

Generate a list of gates for the quantum fourier transform circuit on `n` qubits.
"""
function qft(N::Int; inverse::Bool=false)
  circuit = []
  if inverse
    for j in N:-1:1
      for k in N:-1:(j + 1)
        angle = -π / 2^(k - j)
        push!(circuit, ("CPHASE", (k, j), (ϕ=angle,)))
      end
      push!(circuit, ("H", j))
    end
  else
    for j in 1:(N - 1)
      push!(circuit, ("H", j))
      for k in (j + 1):N
        angle = -π / 2^(k - j)
        push!(circuit, ("CPHASE", (k, j), (ϕ=angle,)))
      end
    end
    push!(circuit, ("H", N))
  end
  return circuit
end

function dft_circuit(s::Vector{<:Index}; inverse::Bool=false)
  return ops(qft(length(s); inverse), s)
end

function dft_matrix(n::Int)
  N = 2^n
  ω = exp(-2π * im / N)
  return [ω^(j * k) / √N for j in 0:(N-1), k in 0:(N-1)]
end

function dft_mpo(s::Vector{<:Index}; alg="mpo", kwargs...)
  return dft_mpo(Algorithm(alg), s; kwargs...)
end

# https://arxiv.org/pdf/2210.08468.pdf
# Eq. 53-54
function phase_mpo(s::Vector{<:Index})
  n = length(s)
  if n == 1
    return MPO(s, "I")
  end
  U = MPO(n)
  l = [Index(2; tags="l=$(n)↔$(n+1)") for n in 1:(n - 1)]
  U[1] = δ(s[1], s[1]', l[1])
  for j in 2:(n - 1)
    U[j] = op("I", s[j]) * onehot(l[j - 1] => 1, l[j] => 1)
    U[j] += op("Phase", s[j]; ϕ=-π/2^(j - 1)) * onehot(l[j - 1] => 2, l[j] => 2)
  end
  U[n] = op("I", s[n]) * onehot(l[n - 1] => 1)
  U[n] += op("Phase", s[n]; ϕ=-π/2^(n - 1)) * onehot(l[n - 1] => 2)
  return U
end

function hadamard_phase_mpo(s::Vector{<:Index})
  U = phase_mpo(s)

  H1 = op("H", s[1])
  U[1] = apply(H1, U[1])
  return U
end

function dft_mpo(::Algorithm"mpo", s::Vector{<:Index}; cutoff=1e-15)
  n = length(s)
  Us = Vector{MPO}(undef, n)
  for j in 1:n
    Ij = if j == 1
      MPO(ITensor[])
    else
      # TODO: Implement for `length(s) == 0`?
      MPO(s[1:(j - 1)], "I")
    end
    Uj = hadamard_phase_mpo(s[j:n])
    Us[j] = [Ij; Uj]
  end
  U = Us[1]
  for j in 2:n
    U = apply(U, Us[j]; cutoff)
  end
  return swapprime(U, 0 => 1)
end

# Discrete Fourier transform MPO from QFT circuit
# Faster to use `alg="mpo"`, not recommended.
function dft_mpo(::Algorithm"circuit", s::Vector{<:Index};
  inverse::Bool=false,
  cutoff=1e-15,
  move_sites_back_between_gates=true,
  move_sites_back=true,
)
  return apply(dft_circuit(s; inverse), MPO(s, "I"); cutoff, move_sites_back_between_gates, move_sites_back)
end

