using ITensors
using ITensorPartialDiffEq
using UnicodePlots

ITensors.disable_warn_order()

include("airy_utils.jl")

results_dirname = "results_more_sweeps"

xᶠ = 20.0
ns = 20:30

# xᶠ = 30.0
# ns = 20:42

# xᶠ = 40.0
# ns = 20:48

us = Vector{MPS}(undef, length(ns))
norm_errors = Vector{Float64}(undef, length(ns))
max_errors = Vector{Float64}(undef, length(ns))
times = Vector{Float64}(undef, length(ns))
for j in eachindex(ns)
  println()

  n = ns[j]
  (; u, xⁱ, α, β, time)  = load_airy_results(; dirname=results_dirname, xᶠ, n)

  N = 2^n
  h = (xᶠ - xⁱ) / (N - 1)

  @show n, N, h

  #
  # Compare to exact result
  #

  nbits_compare = minimum(ns)
  # nbits_compare = 10
  # nbits_compare = min(n, nbits_compare)

  # Maximum of 22, otherwise it is too slow
  # to evaluate the function
  @assert nbits_compare ≤ 22

  u_vec_approx = mps_to_discrete_function(project_bits(u, ones(Int, n - nbits_compare)))
  @show u_vec_approx[1], u_vec_approx[end]
  display(lineplot(u_vec_approx))

  xrange = [xⁱ + n * h for n in (N - 2^nbits_compare):(N - 1)]
  u_vec_exact = u_exact.(xrange, α, β)
  @show u_vec_exact[1], u_vec_exact[end]
  display(lineplot(u_vec_exact))
  display(lineplot(u_vec_exact - u_vec_approx))

  @show number_of_zeros(u_vec_exact)
  @show number_of_zeros(u_vec_approx)

  @show norm(u_vec_exact - u_vec_approx)
  @show maximum(abs, u_vec_exact - u_vec_approx)

  us[j] = u
  norm_errors[j] = norm(u_vec_exact - u_vec_approx)
  max_errors[j] = maximum(abs, u_vec_exact - u_vec_approx)
  times[j] = time
end

xlabel = "log10(sites)"
display(lineplot(ns * log10(2), log10.(norm_errors); title="Norm error", xlabel, ylabel="log10(norm difference)"))
display(lineplot(ns * log10(2), log10.(max_errors); title="Maximum elementwise error", xlabel, ylabel="log10(maximum difference)"))
display(lineplot(ns * log10(2), times; title="Time", xlabel, ylabel="Time (seconds)"))
display(lineplot(ns * log10(2), maxlinkdim.(us); title="Time", xlabel, ylabel="Maximum rank"))

