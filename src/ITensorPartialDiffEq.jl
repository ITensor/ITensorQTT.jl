module ITensorPartialDiffEq

using LinearAlgebra # SymTridiagonal
using ITensors
using ITensorTDVP
using Polynomials
using KrylovKit

using ITensors: Algorithm, @Algorithm_str

include("eigsolve_target.jl")
include("mps_utils.jl")
include("project_bits.jl")
include("function_to_mps.jl")
include("prolongation.jl")
include("laplacian.jl")
include("evolve.jl")
include("integration.jl")
include("dmrg_target.jl")

export function_to_mps,
  mps_to_discrete_function,
  laplacian_mpo,
  integrate_mps,
  prolongate,
  retract,
  sample_bits,
  project_bits,
  dmrg_target,
  interleave

end
