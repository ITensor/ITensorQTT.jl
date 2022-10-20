using ITensors
using ITensorPartialDiffEq
using JLD2
using Plots
using Random
using UnicodePlots

ITensors.disable_warn_order()

include("src/airy_utils.jl")

"""
nxfs = 1:20 # xf in [2^1, 2^2, ..., 2^20]
ns = 1:5 # n in [2^1, 2^2, ..., 2^22]
α, β = 1.0, 1.0 # Boundary conditions `u(xi) = α Ai(-xi) + β Bi(-xi)`, `u(xf) = α Ai(-xf) + β Bi(-xf)`
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solution_compression"
results_dir = joinpath(root_dir, "results")
cutoff = 1e-15 # QTT/MPS compression cutoff
airy_qtt_compression_get_results(nxfs, ns; α, β, results_dir, cutoff)
"""
function airy_qtt_compression_get_results(nxfs, ns; results_dir, α, β, cutoff=1e-15)
  if !isdir(results_dir)
    mkpath(results_dir)
  end
  @time for nxf in nxfs
    @show nxf
    @time for n in ns
      @show n
      results = @time airy_qtt_compression(n, 1.0, 2^nxf; α, β, cutoff)
      jldsave(joinpath(results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2"); results)
    end
  end
end

"""
nxfs = 1:2:11
ns = Dict(1 => 6:22, 3 => 6:22, 5 => 8:22, 7 => 11:22, 9 => 14:22, 11 => 17:22)
best_fit_points = Dict(1 => 6:16, 3 => 6:18, 5 => 8:20, 7 => 12:21, 9 => 18:22, 11 => 20:22)
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solution_compression"
results_dir = joinpath(root_dir, "results")
plots_dir = joinpath(root_dir, "plots")
airy_qtt_compression_plot_results(nxfs, ns; results_dir, plots_dir, best_fit_points)
"""
function airy_qtt_compression_plot_results(nxfs, ns; results_dir, plots_dir, best_fit_points=nothing)
  if !isdir(results_dir)
    error("No results directory $results_dir found")
  end
  if !isdir(plots_dir)
    mkpath(plots_dir)
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
    title="Difference from exact solution",
    legend=:bottomright,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|uᵢ - ũᵢ|²/∑ᵢ|ũᵢ|²",
  )
  plot_airy_error = plot(;
    title="Error satisfying discretized Airy equation",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|(Au)ᵢ - bᵢ|²",
  )
  maxlinkdims = Float64[]
  for nxf in nxfs
    maxlinkdims_nxf = Float64[]
    norm_errors = Float64[]
    airy_errors = Float64[]
    for n in ns[nxf]
      results = load(joinpath(results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2"), "results")
      xi = results.xi
      xf = results.xf
      u = results.u

      # Save maximum the rank of the QTT
      push!(maxlinkdims_nxf, maxlinkdim(u))

      n = length(u)
      u_vec_exact = results.u_vec_exact
      u_vec_approx = mps_to_discrete_function(u)

      # h = (xf - xi) / 2^n
      # ∑ᵢ |uᵢ - ũᵢ|² / ∑ᵢ |ũᵢ|²
      u_diff_integrated = sum(abs2, u_vec_exact - u_vec_approx) / sum(abs2, u_vec_exact)
      push!(norm_errors, u_diff_integrated)

      # How well does it satisfy the Airy equation?
      xf = 2^nxf
      (; A, b) = airy_system(siteinds(u), xi, xf, results.α, results.β)

      push!(airy_errors, abs(sqeuclidean((A, u), b)))
    end
    push!(maxlinkdims, last(maxlinkdims_nxf))
    plot!(plot_norm_error, 2 .^ ns[nxf], norm_errors;
          label="xf=$(2^nxf)",
          linewidth=3,
         )

    # Best fit line
    if !isnothing(best_fit_points)
      nfits = best_fit_points[nxf]
      jfits = [findfirst(==(x), ns[nxf]) for x in nfits]
      a, b = linreg(nfits * log10(2), log10.(airy_errors[jfits]))
      @show nxf, a, b
    end

    plot!(plot_airy_error, 2 .^ ns[nxf], airy_errors;
          label="xf=$(2^nxf)",
          linewidth=3,
         )
  end
  x = 2 .^ nxfs
  a, b = linreg(nxfs * log10(2), log10.(maxlinkdims))
  plot!(plot_maxlinkdim, x, maxlinkdims;
        label="Maximum QTT rank",
        linewidth=3,
       )
  plot!(plot_maxlinkdim, x, 10 ^ a * x .^ b;
        label="Best fit: $(round(10^a; digits=2)) xf ^ $(round(b; digits=2))",
        linewidth=3,
        linestyle=:dash,
       )
  println("Saving plots to $(plots_dir)")
  savefig(plot_maxlinkdim, joinpath(plots_dir, "plot_maxlinkdim.png"))
  savefig(plot_norm_error, joinpath(plots_dir, "plot_norm_error.png"))
  savefig(plot_airy_error, joinpath(plots_dir, "plot_airy_error.png"))
end

# Solve the Airy equation:
#
# u''(x) - x * u(x) = 0
#
# with the boundary values:
#
# u(xi) = ui
# u(xf) = uf
function airy_solver(;
  xi=1.0,
  xf,
  n,
  α,
  β,
  seed=1234,
  init=nothing,
  linsolve_kwargs=(;)
)
  linsolve_kwargs = (;
    nsweeps=10,
    cutoff=1e-15,
    outputlevel=1,
    tol=1e-15,
    krylovdim=30,
    maxiter=30,
    ishermitian=true,
    linsolve_kwargs...,
  )

  # Normalize the coefficients
  α, β = (α, β) ./ norm((α, β))

  @show n, 2^n, n * log10(2)
  @show xi, xf
  @show linsolve_kwargs

  # Starting guess for solution
  if isnothing(init)
    Random.seed!(seed)
    s = siteinds("Qubit", n)
    init = randomMPS(s; linkdims=10)
  else
    s = siteinds(init)
    ## init = replace_siteinds(init, s[1:length(init)])
    ## u = prolongate(init, s[length(init) + 1:end])
  end

  # Set up Au = b
  (; A, b) = airy_system(s, xi, xf, α, β)

  solve_time = @elapsed begin
    # Solve Au = b
    u = linsolve(A, b, init; nsite=2, linsolve_kwargs...)
    # u = linsolve(A, b, u; nsite=1, linsolve_kwargs...)
  end

  @show sqeuclidean_normalized((A, u), b)

  return (; u, init, xi, xf, α, β, seed, linsolve_kwargs, solve_time)
end

"""
# nxfs = 1:2:3 # 1:2:11
# ns = Dict(1 => 6:22, 3 => 6:22, 5 => 8:22, 7 => 11:22, 9 => 14:22, 11 => 17:22)
nxfs = 1:2:11
ns = Dict(1 => 2:22, 3 => 2:2, 5 => 2:22, 7 => 2:22, 9 => 2:22, 11 => 2:22)
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solver"
results_dir = joinpath(root_dir, "results")
airy_solver_run(nxfs, ns; results_dir, save_results=true)
"""
function airy_solver_run(nxfs, ns;
  results_dir,
  save_results,
  multigrid=true, # Use results from shorter system as starting state
  # Other parameters, shouldn't change
  α=1.0,
  β=1.0,
  seed=1234,
  linsolve_kwargs=(;),
)
  linsolve_kwargs = (;
    nsweeps=30,
    cutoff=1e-15,
    outputlevel=1,
    tol=1e-15,
    krylovdim=30,
    maxiter=30,
    ishermitian=true,
    linsolve_kwargs...
  )
  @show linsolve_kwargs

  if !isdir(results_dir)
    @warn "Results directory $results_dir doesn't exist, making it now."
    mkpath(results_dir)
  end

  for nxf in nxfs
    xf = 2^nxf
    for n in ns[nxf]
      println("\n" * "#"^100)
      println("Running Airy equation QTT solver for `xf = $(xf)` and `n = $(n)`.\n")
      u_init = nothing
      if multigrid
        println("\nTry using state from length $(n - 1) as starting state.")
        init_filename = airy_solver_filename(; dirname=results_dir, xf=xf, n=(n - 1))
        if !isfile(init_filename)
          @warn "File $init_filename doesn't exist, a random starting state will be used instead."
        else
          u_init = load_airy_results(; dirname=results_dir, xf=xf, n=(n - 1)).u
          u_init = prolongate(u_init, siteind("Qubit", n))
        end
      else
        @warn "Not saving results to file $filename."
      end
      result = airy_solver(; xf, n, α, β, seed, init=u_init, linsolve_kwargs)
      filename = airy_solver_filename(; dirname=results_dir, xf, n)
      if save_results
        println("\nSaving results to file $filename.")
        if isfile(filename)
          @warn "File $filename already exists, overwriting."
        end
        jldsave(filename; result...)
      else
        @warn "Not saving results to file $filename."
      end
    end
  end
end

"""
# nxfs = 1:2:3 # 1:2:11
# ns = Dict(1 => 6:22, 3 => 6:22, 5 => 8:22, 7 => 11:22, 9 => 14:22, 11 => 17:22)
nxfs = 1:2:3 # 1:2:11
ns = Dict(1 => 2:10, 3 => 2:10, 5 => 7:22, 7 => 9:22, 9 => 11:22, 11 => 13:22)
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solver"
results_dir = joinpath(root_dir, "results")
exact_results_dir = joinpath(root_dir, "..", "airy_solution_compression", "results")
plots_dir = joinpath(root_dir, "plots")
airy_solver_analyze(nxfs, ns; results_dir, exact_results_dir, plots_dir)
"""
function airy_solver_analyze(nxfs, ns; results_dir, exact_results_dir, plots_dir)
  if !isdir(results_dir)
    error("No results directory $(results_dir)")
  end
  if !isdir(plots_dir)
    @warn "Making the directory path $(plots_dir)"
    mkpath(plots_dir)
  end
  error_linsolves = Dict()
  error_diffs = Dict()
  solve_times = Dict()
  maxlinkdims = Dict()
  for nxf in nxfs
    @show nxf
    error_linsolves[nxf] = Float64[]
    error_diffs[nxf] = Float64[]
    solve_times[nxf] = Float64[]
    maxlinkdims[nxf] = Float64[]
    for n in ns[nxf]
      println()
      @show nxf, n

      # results = load(airy_solver_filename(; dirname=results_dir, xf=2^nxf, n=n), "results")
      results = load_airy_results(; dirname=results_dir, xf=2^nxf, n=n)
      u = results.u
      α = results.α
      β = results.β
      α, β = (α, β) ./ norm((α, β))

      @show norm(u)

      xi = results.xi
      xf = results.xf
      solve_time = results.solve_time

      s = siteinds(u)
      n = length(u)
      N = 2^n
      h = (xf - xi) / N
      (; A, b) = airy_system(s, xi, xf, α, β)

      error_linsolve = @show sqeuclidean_normalized((A, u), b)
      if iszero(error_linsolve)
        error_linsolve = 1e-15
      end

      # TODO: Implement airy_compression_filename(nxf, n))
      exact_results = load(joinpath(exact_results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2"), "results")
      u_exact = exact_results.u
      u_exact = replace_siteinds(u_exact, s)

      if n ≤ 10
        display(lineplot(mps_to_discrete_function(u)))
        display(lineplot(mps_to_discrete_function(u_exact)))
      end

      @show norm(u_exact), norm(u)
      error_diff = @show sqeuclidean_normalized(u, u_exact)

      push!(error_linsolves[nxf], error_linsolve)
      push!(error_diffs[nxf], error_diff)
      push!(solve_times[nxf], solve_time)
      push!(maxlinkdims[nxf], maxlinkdim(u))
    end
  end

  # Make plots
  plot_error_linsolves = plot(;
    title="Airy QTT solver",
    legend=:topright,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|Auᵢ - bᵢ|²",
  )
  plot_error_diffs = plot(;
    title="Airy QTT solver",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|uᵢ - ũᵢ|² / ∑ᵢ|ũᵢ|²",
  )
  plot_maxlinkdims = plot(;
    title="Airy QTT solver",
    legend=:topleft,
    xaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="QTT Rank",
  )
  plot_time = plot(;
    title="Airy QTT solver",
    legend=:topleft,
    xaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="Time to solve (seconds)",
  )
  for nxf in nxfs
    plot!(plot_error_linsolves, 2 .^ ns[nxf], abs.(error_linsolves[nxf]);
      label="xf = 10^$(round(nxf * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_error_diffs, 2 .^ ns[nxf], abs.(error_diffs[nxf]);
      label="xf = 10^$(round(nxf * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_maxlinkdims, 2 .^ ns[nxf], maxlinkdims[nxf];
      label="xf = 10^$(round(nxf * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_time, 2 .^ ns[nxf], solve_times[nxf];
      label="xf = 10^$(round(nxf * log10(2); digits=2))",
      linewidth=3,
    )
  end

  println("Saving plots to $(plots_dir)")
  Plots.savefig(plot_error_linsolves, joinpath(plots_dir, "plot_error_linsolves.png"))
  Plots.savefig(plot_error_diffs, joinpath(plots_dir, "plot_error_diffs.png"))
  Plots.savefig(plot_maxlinkdims, joinpath(plots_dir, "plot_qtt_rank.png"))
  Plots.savefig(plot_time, joinpath(plots_dir, "plot_solve_time.png"))
  return nothing
end
