using ITensors
using ITensorPartialDiffEq
using Plots
using UnicodePlots

ITensors.disable_warn_order()

include("src/airy_utils.jl")

function getindex_qtt(u, I::StepRange{Int,Int})
  return [getindex_qtt(u, i) for i in I]
end

function getindex_qtt(u, i::Int)
  n = length(u)
  bits = reverse(digits(i - 1, base=2, pad=n))
  return only(project_bits(u, bits))[]
end

# results_dirname = "results_more_sweeps"
# xᶠs = [20.0, 30.0, 40.0]
# nss = Dict(20.0 => 10:30, 30.0 => 10:42, 40.0 => 10:48)

results_dirname = "results_multigrid"
xᶠs = [2.0, 10.0, 30.0]
nss = Dict(2.0 => 10:20, 10.0 => 10:13, 30.0 => 10:20)

# comparison = "begin"
# comparison = "end"
comparison = "grid"

# Results
us = Dict()
norm_errors = Dict()
max_errors = Dict()
times = Dict()

# Number of points to compare
ncompare = 10

for xᶠ in xᶠs
  ns = nss[xᶠ]
  us[xᶠ] = Vector{MPS}(undef, length(ns))
  norm_errors[xᶠ] = Vector{Float64}(undef, length(ns))
  max_errors[xᶠ] = Vector{Float64}(undef, length(ns))
  times[xᶠ] = Vector{Float64}(undef, length(ns))
  for j in eachindex(ns)
    println()

    n = ns[j]
    (; u, xⁱ, α, β, time)  = load_airy_results(; dirname=results_dirname, xᶠ, n)
    u /= norm(u)

    N = 2^n

    Nstep = 2^(n - ncompare)
    compare_range = 1:Nstep:N

    # @show collect(compare_range)
    @show N, Nstep, N - Nstep

    # h = (xᶠ - xⁱ) / (N - 1)
    h = (xᶠ - xⁱ) / (N - 1)

    @show n, N, h

    #
    # Compare to exact result
    #

    # Maximum of 22, otherwise it is too slow
    # to evaluate the function
    # @assert nbits_compare ≤ 22

    if comparison == "begin"
      # Compare the last section of the vector
      u_vec_approx = mps_to_discrete_function(project_bits(u, zeros(Int, n - ncompare)))
      xrange = [xⁱ + n * h for n in 0:(2^ncompare - 1)]
    elseif comparison == "end"
      # Compare the last section of the vector
      u_vec_approx = mps_to_discrete_function(project_bits(u, ones(Int, n - ncompare)))
      xrange = [xⁱ + n * h for n in (N - 2^ncompare):(N - 1)]
    elseif comparison == "grid"
      # Compare on an equally spaced grid
      u_vec_approx = getindex_qtt(u, compare_range)
      # xrange = xⁱ .+ (compare_range .- 1) .* h
      xrange = xⁱ .+ (xᶠ - xⁱ) * (compare_range .- 1) ./ N
    else
      error("No comparison $comparison")
    end

    u_vec_approx /= norm(u_vec_approx)
    @show u_vec_approx[1], u_vec_approx[end]
    display(lineplot(u_vec_approx))

    @show xrange[end]
    @show (2 * xrange[end] - xrange[end - 1]), xᶠ

    @show xⁱ, xᶠ, h
    # @show xrange

    u_vec_exact = airy_solution.(xrange, α, β)
    u_vec_exact /= norm(u_vec_exact)
    @show u_vec_exact[1], u_vec_exact[end]
    display(lineplot(u_vec_exact))
    display(lineplot(u_vec_exact - u_vec_approx))

    @show number_of_zeros(u_vec_exact)
    @show number_of_zeros(u_vec_approx)

    @show norm(u_vec_exact - u_vec_approx)
    @show norm(u_vec_exact - u_vec_approx) / length(u_vec_approx)
    @show maximum(abs, u_vec_exact - u_vec_approx)

    us[xᶠ][j] = u
    norm_errors[xᶠ][j] = norm(u_vec_exact - u_vec_approx) / length(u_vec_approx)
    max_errors[xᶠ][j] = maximum(abs, u_vec_exact - u_vec_approx)
    times[xᶠ][j] = time
  end
end

xlabel = "log10(sites)"

p = plot()
for xᶠ in xᶠs
  plot_kwargs = (;
    title="Norm error",
    xlabel,
    ylabel="log10(norm difference)",
    label="xf = $(xᶠ)",
  )
  plot!(p, nss[xᶠ] * log10(2), log10.(norm_errors[xᶠ]); plot_kwargs...)
end
plots_dir = joinpath(results_dirname, "plots")
if !isdir(plots_dir)
  mkdir(plots_dir)
end
Plots.savefig(p, joinpath(plots_dir, "airy_xf_$(xᶠs)_norm_errors.png"))
