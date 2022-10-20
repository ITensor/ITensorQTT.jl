function project_bits(ψ::MPS, bits::Vector)
  s = siteinds(ψ)
  nbits = length(ψ)
  nproj = length(bits)
  proj = ITensor(1.0)
  for j in 1:nproj
    proj = proj * (onehot(s[j] => bits[j] + 1) * ψ[j])
  end
  ψ_proj = MPS(ψ[(nproj + 1):nbits])
  if iszero(length(ψ_proj))
    return proj[]
  end
  ψ_proj[1] *= proj
  return ψ_proj
end

function sample_bits(ψ::MPS, bits=1:length(ψ))
  return (sample(ψ) .- 1)[bits]
end

function project_bits(ψ::MPS, bits::Tuple{Vararg{Vector}})
  return project_bits(ψ, interleave(bits...))
end

function project_bits(u::MPS, left_bits::Vector{Int}, right_bits::Vector{Int})
  u = project_bits(u, left_bits)
  return reverse(project_bits(reverse(u), reverse(right_bits)))
end

function project_bits(u::MPS, left_bits::Vector{Int}, nright_bits::Int)
  return project_bits(u, left_bits, fill(0, nright_bits))
end
