using ITensors
using ITensorQTT
using JLD2
using Plots

using ITensorQTT: vec_to_mps

ITensors.disable_warn_order()

include("src/airy_utils.jl")

function airy_qtt(s::Vector{<:Index}, xi, xf; α=1.0, β=0.0, cutoff=1e-15)
  α, β = (α, β) ./ norm((α, β))
  return function_to_mps(x -> airy_solution(x, α, β), s, xi, xf; cutoff)
end

function airy_qtt_error(n::Int, xi, xf; α=1.0, β=0.0, cutoff=1e-15)
  α, β = (α, β) ./ norm((α, β))

  # Exact Airy solution from SpecialFunctions.jl
  xrange = qtt_xrange(n, xi, xf)
  u_vec_exact = airy_solution.(xrange, α, β)

  s = siteinds("Qubit", n)
  u = vec_to_mps(u_vec_exact, s; cutoff)

  return (; u, u_vec_exact, α, β, cutoff, xi, xf)
end

# ns = 1:22
# nxfs = 1:20
# α, β = 1, 1
# airy_qtt_compression_get_results(1:20, 1:22; α=1.0, β=1.0, results_dir="results", cutoff=1e-15)
function airy_qtt_compression_get_results(nxfs, ns; results_dir, α, β, cutoff=1e-15)
  if !isdir(results_dir)
    mkdir(results_dir)
  end
  @time for nxf in nxfs
    @show nxf
    @time for n in ns
      @show n
      results = @time airy_qtt_error(n, 1.0, 2^nxf; α, β, cutoff)
      jldsave(
        joinpath(results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2");
        results,
      )
    end
  end
end

# nxfs = 1:2:11
# ns = Dict(1 => 6:22, 3 => 6:22, 5 => 8:22, 7 => 11:22, 9 => 14:22, 11 => 17:22)
# airy_qtt_compression_plot_results(nxfs, ns; results_dir="results", plots_dir="plots")
function airy_qtt_compression_plot_results(nxfs, ns; results_dir, plots_dir)
  if !isdir(results_dir)
    error("No results directory $results_dir found")
  end
  if !isdir(plots_dir)
    mkdir(plots_dir)
  end
  plot_maxlinkdim = plot(;
    title="Maximum QTT rank",
    legend=:topleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="xf",
    ylabel="Maximum QTT rank",
  )
  plot_norm_error = plot(;
    title="Integral of difference from exact solution",
    legend=:bottomright,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∫|u(x) - ũ(x)|²dx / (xf - xi)",
  )
  plot_airy_error = plot(;
    title="Error from satisfying Airy equation",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="|Au⃗ - b⃗|",
  )
  maxlinkdims = Float64[]
  for nxf in nxfs
    maxlinkdims_nxf = Float64[]
    norm_errors = Float64[]
    airy_errors = Float64[]
    for n in ns[nxf]
      results = load(
        joinpath(results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2"),
        "results",
      )
      xi = results.xi
      xf = results.xf
      u = results.u

      # Save maximum the rank of the QTT
      push!(maxlinkdims_nxf, maxlinkdim(u))

      n = length(u)
      u_vec_exact = results.u_vec_exact
      u_vec_approx = mps_to_discrete_function(u)

      # h = (xf - xi) / 2^n
      # ∑ᵢ |uᵢ - ũᵢ|² / 2^n
      # Normalized by `(xf - xi)`
      u_diff_integrated = sum(abs2, u_vec_exact - u_vec_approx) / 2^n
      push!(norm_errors, u_diff_integrated)

      # How well does it satisfy the Airy equation?
      xi = 1.0
      xf = 2^nxf
      (; A, b) = airy_system(siteinds(u), xi, xf, results.α, results.β)
      push!(airy_errors, linsolve_error(A, u, b))
    end
    push!(maxlinkdims, last(maxlinkdims_nxf))
    plot!(plot_norm_error, 2 .^ ns[nxf], norm_errors; label="xf=$(2^nxf)", linewidth=3)
    plot!(plot_airy_error, 2 .^ ns[nxf], airy_errors; label="xf=$(2^nxf)", linewidth=3)
  end
  x = 2 .^ nxfs
  a, b = linreg(nxfs * log10(2), log10.(maxlinkdims))
  plot!(plot_maxlinkdim, x, maxlinkdims; label="Maximum QTT rank", linewidth=3)
  plot!(
    plot_maxlinkdim,
    x,
    10^a * x .^ b;
    label="Best fit: $(round(10^a; digits=2)) xf ^ $(round(b; digits=2))",
    linewidth=3,
    linestyle=:dash,
  )
  savefig(plot_maxlinkdim, joinpath(plots_dir, "plot_maxlinkdim.png"))
  savefig(plot_norm_error, joinpath(plots_dir, "plot_norm_error.png"))
  return savefig(plot_airy_error, joinpath(plots_dir, "plot_airy_error.png"))
end
