using ITensors
using ITensorPartialDiffEq
using JLD2
using Plots
using Random
using UnicodePlots
using FFTW

ITensors.disable_warn_order()

include("src/airy_utils.jl")

"""
ns = 7:14
xi, xf = 1.0, 5.0
α, β = 1.0, 1.0
res = airy_compare_to_matrix.(ns; α, β, xi, xf)
airy_err = first.(res)
integral_err = last.(res)
linreg(ns, log2.(airy_err)) # ∫dx |A(x,x')u(x') - b(x)|
# 2-element Vector{Float64}:
#   0.6864681664612592
#  -1.997374659387574
linreg(ns, log2.(integral_err)) # ∫dx |u(x) - ũ(x)|²
# 2-element Vector{Float64}:
#   1.1385702262631532
#  -1.9818952100419311

# Resource scaling with constant integral error `∫dx |u(x) - ũ(x)|²`:
# N = 2 * xf²
# log₂(N) = n = log₂(2 * xf²) = log₂(xf²) + log₂(2) = 2 * log₂(xf) + 1 = 2nxf + 1
nxfs = 1:8 #1:12
res = @time [airy_compare_to_matrix(2nxf + 1; α, β, xi, xf=2^nxf, fft_cutoff=3.0) for nxf in nxfs]
airy_err = getindex.(res, :linsolve_error)
integral_err = getindex.(res, :diff)

# Fourier transform results
Ns = log2.(getindex.(res, :N))
nnz_ffts = log2.(getindex.(res, :nnz_fft))
linreg(Ns, nnz_ffts)
lineplot(Ns, nnz_ffts)
"""
function airy_compare_to_matrix(n; α, β, xi, xf, fft_cutoff=1.0)
  α, β = (α, β) ./ norm((α, β))
  N = 2^n
  s = siteinds("Qubit", n)

  h = (xf - xi) / N
  # xs = range(; start=xi, stop=xf, length=(N+1))[1:N]
  xs = [xi + j * h for j in 0:(N - 1)]

  println("Compute discrete Airy function")
  u_exact = @time airy_solution.(xs, α, β)
  u_fft = fft(u_exact)
  nnz_fft = @show count(x -> abs(x) > fft_cutoff, u_fft)

  Ab_mpo = airy_system(s, xi, xf, α, β)

  if n < 15
    A_mpo = @time mpo_to_mat(Ab_mpo.A)
  end
  b_mpo = mps_to_discrete_function(Ab_mpo.b)
  if n < 15
    u_mpo = A_mpo \ b_mpo
  end

  println("Solve Airy equation with matrix A\\b")
  Ab_mat = airy_system_matrix(s, xi, xf, α, β)
  A_mat = Ab_mat.A
  b_mat = Ab_mat.b
  u_mat = @time A_mat \ b_mat

  display(lineplot(u_exact; label="u_exact"))

  if n < 15
    println("From MPO")
    display(A_mpo)
    display(lineplot(u_mpo; label="u_mpo"))
  end

  println("Matrix version")
  display(A_mat)
  display(lineplot(u_mat; label="u_mat"))

  display(lineplot(log10.(abs2.(u_mat - u_exact)); label="log10.(|u_mat - u_exact|²)"))

  @show n, N
  @show sum(abs2, u_mat - u_exact) / N
  if n < 15
    @show sum(abs2, u_mpo - u_exact) / N
  end
  @show norm(A_mat * u_exact - b_mat)
  @show sum(abs, A_mat * u_exact - b_mat) / N
  @show norm(A_mat * u_mat - b_mat)

  @show number_of_zeros(u_exact)
  if n < 15
    @show number_of_zeros(u_mpo)
  end
  @show number_of_zeros(u_mat)

  if n < 15
    @show diff(diag(A_mpo) / h^2)[1:min(N - 1, 10)]
  end
  @show diff(diag(A_mat) / h^2)[1:min(N - 1, 10)]
  @show (xf - xi) / (N - 1)
  @show (xf - xi) / N

  @show norm(b_mpo - b_mat)
  if n < 15
    @show norm(A_mpo - A_mat)
  end
  return (; nnz_fft, N, linsolve_error=sum(abs, A_mat * u_exact - b_mat) / N, diff=sum(abs2, u_mat - u_exact) / N)
end

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
  labelfontsize = 12
  tickfontsize = 10
  legendfontsize = 10
  if !isdir(results_dir)
    error("No results directory $results_dir found")
  end
  if !isdir(plots_dir)
    mkpath(plots_dir)
  end
  plot_qtt_rank = plot(;
    title="Maximum QTT rank",
    legend=:topleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="xf",
    ylabel="QTT rank",
    xtickfontsize=tickfontsize,
    ytickfontsize=tickfontsize,
    xguidefontsize=labelfontsize,
    yguidefontsize=labelfontsize,
    legendfontsize,
  )
  plot_memory = plot(;
    title="QTT memory usage",
    legend=:topleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="xf",
    ylabel="Memory usage",
    xtickfontsize=tickfontsize,
    ytickfontsize=tickfontsize,
    xguidefontsize=labelfontsize,
    yguidefontsize=labelfontsize,
    legendfontsize,
  )
  plot_norm_error = plot(;
    title="Difference from exact solution",
    legend=:bottomright,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|uᵢ - ũᵢ|²/∑ᵢ|ũᵢ|²",
    xtickfontsize=tickfontsize,
    ytickfontsize=tickfontsize,
    xguidefontsize=labelfontsize,
    yguidefontsize=labelfontsize,
    legendfontsize,
  )
  plot_airy_error = plot(;
    title="Error satisfying discretized Airy equation",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|(Au)ᵢ - bᵢ|²",
    xtickfontsize=tickfontsize,
    ytickfontsize=tickfontsize,
    xguidefontsize=labelfontsize,
    yguidefontsize=labelfontsize,
    legendfontsize,
  )
  plot_airy_exact_error = plot(;
    title="Error of exact solution from satisfying discretized Airy equation",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|(Au)ᵢ - bᵢ|²",
    xtickfontsize=tickfontsize,
    ytickfontsize=tickfontsize,
    xguidefontsize=labelfontsize,
    yguidefontsize=labelfontsize,
    legendfontsize,
  )
  maxlinkdims = Float64[]
  maxveclengths = Float64[]
  for nxf in nxfs
    maxlinkdims_nxf = Float64[]
    norm_errors = Float64[]
    airy_errors = Float64[]
    airy_exact_errors = Float64[]
    for n in ns[nxf]
      results = load(joinpath(results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2"), "results")
      xi = results.xi
      xf = results.xf
      u = results.u

      # Save maximum the rank of the QTT
      @show maxlinkdim(u)
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

      Ab_mat = airy_system_matrix(siteinds(u), xi, xf, results.α, results.β)
      A_mat = Ab_mat.A
      b_mat = Ab_mat.b

      push!(airy_errors, abs(sqeuclidean((A, u), b)))
      push!(airy_exact_errors, sum(abs2, A_mat * u_vec_exact - b_mat))
    end

    error_threshold = 1e-7
    j_constant_error = findfirst(<(error_threshold), airy_errors)
    n_constant_error = @show ns[nxf][j_constant_error]
    push!(maxveclengths, 2^(n_constant_error))
    push!(maxlinkdims, maxlinkdims_nxf[j_constant_error])

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

    plot!(plot_airy_exact_error, 2 .^ ns[nxf], airy_exact_errors;
          label="xf=$(2^nxf), exact",
          linewidth=3,
         )
  end
  x = 2 .^ nxfs

  # Plot QTT rank
  plot!(plot_qtt_rank, x, maxlinkdims;
        label="QTT",
        linewidth=3,
       )

  a_qtt_rank, b_qtt_rank = linreg(nxfs * log10(2), log10.(maxlinkdims))
  plot!(plot_qtt_rank, x, 10 ^ a_qtt_rank * x .^ b_qtt_rank;
        label="Best fit: $(round(10^a_qtt_rank; digits=2)) xf ^ $(round(b_qtt_rank; digits=2))",
        linewidth=3,
        linestyle=:dash,
       )

  # Plot memory usage
  plot!(plot_memory, x, 2 .* log2.(maxveclengths) .* maxlinkdims .^ 2;
        label="QTT",
        linewidth=3,
       )

  a_qtt, b_qtt = linreg(nxfs * log10(2), log10.(2 .* log2.(maxveclengths) .* maxlinkdims .^ 2))
  plot!(plot_memory, x, 10 ^ a_qtt * x .^ b_qtt;
        label="Best fit: $(round(10^a_qtt; digits=2)) xf ^ $(round(b_qtt; digits=2))",
        linewidth=3,
        linestyle=:dash,
       )

  plot!(plot_memory, x, maxveclengths;
        label="Number of gridpoints",
        linewidth=3,
       )

  a_fd, b_fd = linreg(nxfs * log10(2), log10.(maxveclengths))
  plot!(plot_memory, x, 10 ^ a_fd * x .^ b_fd;
        label="Best fit: $(round(10^a_fd; digits=2)) xf ^ $(round(b_fd; digits=2))",
        linewidth=3,
        linestyle=:dash,
       )

  println("Saving plots to $(plots_dir)")
  Plots.savefig(plot_qtt_rank, joinpath(plots_dir, "airy_qtt_rank.pdf"))
  Plots.savefig(plot_memory, joinpath(plots_dir, "airy_memory.pdf"))
  Plots.savefig(plot_norm_error, joinpath(plots_dir, "airy_error_diffs.pdf"))
  Plots.savefig(plot_airy_error, joinpath(plots_dir, "airy_error_diffeq.pdf"))
  Plots.savefig(plot_airy_exact_error, joinpath(plots_dir, "airy_exact_error_diffeq.pdf"))
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
  variant="pseudoinverse",
  linsolve_kwargs=(;),
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
    if variant == "pseudoinverse"
      u = ITensorPartialDiffEq.linsolve_pseudoinverse(A, b, init; nsite=2, linsolve_kwargs...)
    elseif variant == "b_basis"
      u = b_linsolve(A, b, init; nsite=2, linsolve_kwargs..., ishermitian=false)
    elseif isnothing(variant)
      u = linsolve(A, b, init; nsite=2, linsolve_kwargs...)
    else
      error("linsolve variant $variant not supported")
    end
    # u = linsolve(A, b, u; nsite=1, linsolve_kwargs...)
  end

  @show sqeuclidean_normalized((A, u), b)

  return (; u, init, xi, xf, α, β, seed, linsolve_kwargs, solve_time)
end

"""
variant = "pseudoinverse" # [nothing, "pseudoinverse", "b_basis"]
nxfs = 11:11 # 1:2:9
ns = Dict([nxf => 2:22 for nxf in nxfs])
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solver"
if !isnothing(variant)
  root_dir *= "_" * variant
end
results_dir = joinpath(root_dir, "results")
airy_solver_run(nxfs, ns; results_dir, save_results=true, variant)
"""
function airy_solver_run(nxfs, ns;
  results_dir,
  save_results,
  multigrid=true, # Use results from shorter system as starting state
  variant="pseudoinverse",
  # Other parameters, shouldn't change
  α=1.0,
  β=1.0,
  seed=1234,
  linsolve_kwargs=(;),
)
  linsolve_kwargs = (;
    nsweeps=12,
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
          @show maxlinkdim(u_init)
          u_init = prolongate(u_init, siteind("Qubit", n); cutoff=1e-15)
          @show maxlinkdim(u_init)
        end
      else
        @warn "Not saving results to file $filename."
      end
      result = airy_solver(; xf, n, α, β, seed, init=u_init, variant, linsolve_kwargs)
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
variant = "pseudoinverse" # [nothing, "pseudoinverse", "b_basis"]
nxfs = 1:2:9
ns = Dict([nxf => 2:22 for nxf in nxfs])
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solver"
if !isnothing(variant)
  root_dir *= "_" * variant
end
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

      # TODO: Implement airy_compression_filename(nxf, n)
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
    yaxis=:log,
    linewidth=3,
    xlabel="xf",
    ylabel="Time to solve (seconds)",
  )
  for nxf in nxfs
    plot!(plot_error_linsolves, 2 .^ ns[nxf], abs.(error_linsolves[nxf]);
      label="xf = $(2^nxf)",
      linewidth=3,
    )
    plot!(plot_error_diffs, 2 .^ ns[nxf], abs.(error_diffs[nxf]);
      label="xf = $(2^nxf)",
      linewidth=3,
    )
    plot!(plot_maxlinkdims, 2 .^ ns[nxf], maxlinkdims[nxf];
      label="xf = $(2^nxf)",
      linewidth=3,
    )
  end

  x = 2 .^ nxfs
  y = [last(solve_times[nxf]) for nxf in nxfs]
  plot!(plot_time, x, y;
    label="Time to solve",
    linewidth=3,
  )

  a, b = linreg(nxfs * log10(2), log10.(y))
  plot!(plot_time, x, 10 ^ a * (2 .^ (nxfs * b));
        label="Best fit: $(round(10^a; digits=2)) xf ^ $(round(b; digits=2))",
        linewidth=3,
        linestyle=:dash,
       )

  println("Saving plots to $(plots_dir)")
  Plots.savefig(plot_error_linsolves, joinpath(plots_dir, "airy_solver_error_linsolves.pdf"))
  Plots.savefig(plot_error_diffs, joinpath(plots_dir, "airy_solver_error_diffs.pdf"))
  Plots.savefig(plot_maxlinkdims, joinpath(plots_dir, "airy_solver_qtt_rank.pdf"))
  Plots.savefig(plot_time, joinpath(plots_dir, "airy_solver_solve_time.pdf"))
  return nothing
end

"""
variant = "pseudoinverse" # [nothing, "pseudoinverse", "b_basis"]
nk = 18
n = 32 # 28:34
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solver"
if !isnothing(variant)
  root_dir *= "_" * variant
end
results_dir = joinpath(root_dir, "results")
exact_results_dir = joinpath(root_dir, "..", "airy_solution_compression", "results")
airy_solver_visualize_solution(nk, n, 0.75, -1; results_dir, exact_results_dir, plots_dir)
"""
function airy_solver_visualize_solution(; xf::Int, n::Int, xstart::Float64, zoom::Int=0, kwargs...)
  # nzoom is the zoom level relative to the length scale `xf`
  nproj = Int(log2(xf)) + zoom

  # Assume range [xi, xf) = [0.0, 1.0)

  if xstart ≥ 1
    # Catch overflow
    proj = fill(1, n)
  else
    proj = reverse(digits(Int(round(xstart * 2^n)); base=2, pad=n))
  end

  @show proj
  @show nproj
  #nproj = min(length(proj), nproj)
  @show nproj

  proj = proj[1:nproj]

  @show proj

  return airy_solver_visualize_solution(xf, n, proj; kwargs...)
end

function airy_solver_visualize_solution(xf::Int, n::Int, proj::Vector{Int}; results_dir, exact_results_dir, nplot=min(10, n))
  nxf = Int(log2(xf))

  nplot = min(n, nplot)

  if nplot + length(proj) > n
    nplot = n - length(proj)
  end

  nproj = length(proj)
  xstart = sum([proj[j] * 2.0^(-j) for j in 1:nproj])
  xstop = xstart + sum([2.0^(-j) for j in (nproj + 1):n])
  xrange = range(; start=xstart, stop=xstop, length=2^nplot)

  # x̃ = αx + β
  # α = (x̃f - x̃i) / (xf - xi)
  # β = x̃i - (x̃f - x̃i) / (xf - xi) * xi
  #
  # xi = 0, xf = 1
  # x̃i = 1, x̃f = xf
  # x̃ = (xf - 1) * x + 1
  xrange = (xf - 1) .* xrange .+ 1

  @show nxf, n

  # results = load(joinpath(results_dir, airy_solver_filename(xf, n)), "results")
  results = load_airy_results(; dirname=results_dir, xf, n)
  u = results.u
  s = siteinds(u)

  exact_results = load(joinpath(exact_results_dir, "airy_qtt_compression_xi_1.0_xf_2^$(nxf)_n_$(n).jld2"), "results")
  u_exact = exact_results.u

  @show length(u)
  @show length(u_exact)

  u_exact = replace_siteinds(u_exact, s)

  @show sqeuclidean_normalized(u, u_exact)
  @show sqeuclidean(u, u_exact)

  @show norm(u)
  @show norm(u_exact)

  nleft = length(proj)
  nright = n - nplot - nleft

  # @show nplot, nleft, nright

  left_bits = proj

  @show left_bits
  @show nright

  @show maxlinkdim(u)
  @show maxlinkdim(u_exact)

  u_plot = project_bits(u, left_bits, nright)
  u_exact_plot = project_bits(u_exact, left_bits, nright)

  u_vec = mps_to_discrete_function(u_plot)
  u_exact_vec = mps_to_discrete_function(u_exact_plot)

  return (; xrange, u_vec, u_exact_vec)
end

"""
variant = "pseudoinverse" # [nothing, "pseudoinverse", "b_basis"]
nxfs = 1:2:9
ns = 20:22
xstarts = 0:0.5:1.0
zoom = -2
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/airy_solver"
if !isnothing(variant)
  root_dir *= "_" * variant
end
results_dir = joinpath(root_dir, "results")
exact_results_dir = joinpath(root_dir, "..", "airy_solution_compression", "results")
plots_dir = joinpath(root_dir, "plots")
airy_solver_plot_solutions(2 .^ nxfs, ns, xstarts, zoom; results_dir, exact_results_dir, plots_dir)
"""
function airy_solver_plot_solutions(xfs, ns, xstarts, zoom; results_dir, exact_results_dir, plots_dir)
  for xf in xfs
    for xstart in xstarts
      plot_u_diff = plot(;
        title="Airy solution error, xf = $(xf)",
        legend=:bottomright,
        linewidth=3,
        xlabel="x",
        ylabel="|u(x) - ũ(x)|",
        yaxis=:log,
        xformatter=:plain, # disable scientific notation
        yrange=[1e-8, 1e-2],
      )
      for n in ns
        (; xrange, u_vec, u_exact_vec) = airy_solver_visualize_solution(; xf, n, xstart, zoom, results_dir, exact_results_dir)

        # Plot the original functions
        plot_u = plot(;
          title="Airy solution, xf = $(xf), n = 2^$(n)",
          legend=:topleft,
          linewidth=3,
          xlabel="x",
          ylabel="u(x)",
          xformatter=:plain, # disable scientific notation
        )
        plot!(plot_u, xrange, u_vec;
          label="QTT solution",
          linewidth=3,
        )
        plot!(plot_u, xrange, u_exact_vec;
          label="Airy function",
          linewidth=3,
        )
        Plots.savefig(plot_u, joinpath(plots_dir, "airy_visualize_xf_$(xf)_n_$(n)_xstart_$(xstart)_zoom_$(zoom)_qtt.pdf"))

        # Plot the error
        plot!(plot_u_diff, xrange, abs.(u_vec - u_exact_vec);
          label="2^$n gridpoints",
          linewidth=3,
        )
      end
      Plots.savefig(plot_u_diff, joinpath(plots_dir, "airy_visualize_xf_$(xf)_ns_$(ns)_xstart_$(xstart)_zoom_$(zoom)_diff.pdf"))
    end
  end
end
