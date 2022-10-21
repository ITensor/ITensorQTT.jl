module ITensorPartialDiffEq

using LinearAlgebra # SymTridiagonal
using ITensors
using ITensorTDVP
using Polynomials
using KrylovKit

using ITensors: Algorithm, @Algorithm_str

include("utils.jl")
include("eigsolve_target.jl")
include("linsolve.jl")
include("mps_utils.jl")
include("qtt_utils.jl")
include("project_bits.jl")
include("function_to_mps.jl")
include("prolongation.jl")
include("laplacian.jl")
include("evolve.jl")
include("integration.jl")
include("dmrg_target.jl")

export function_to_mps,
  qtt_xrange,
  mps_to_discrete_function,
  mpo_to_mat,
  laplacian_mpo,
  integrate_mps,
  prolongate,
  retract,
  sample_bits,
  project_bits,
  dmrg_target,
  linsolve,
  interleave,
  boundary_value_mps,
  boundary_value_vector,
  number_of_zeros,
  linreg,
  insert_missing_links,
  rescale,
  qtt,
  sqeuclidean,
  sqeuclidean_normalized,
  vec_to_mps

end
