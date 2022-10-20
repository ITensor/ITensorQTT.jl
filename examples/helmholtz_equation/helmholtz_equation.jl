using ITensors
using ITensorPartialDiffEq
using JLD2
using Plots

using ITensorPartialDiffEq: insert_missing_links

ITensors.disable_warn_order()

function ITensorPartialDiffEq.project_bits(u::MPS, left_bits::Vector{Int}, right_bits::Vector{Int})
  u = project_bits(u, left_bits)
  return reverse(project_bits(reverse(u), reverse(right_bits)))
end

function ITensorPartialDiffEq.project_bits(u::MPS, left_bits::Vector{Int}, nright_bits::Int)
  return project_bits(u, left_bits, fill(0, nright_bits))
end

"""
Find `α, β` such that `x̃ = α * x + β` given
`(xi, x̃i)` and `(xf, x̃f)`.
"""
function rescale(xi, xf, x̃i, x̃f)
  α = (x̃f - x̃i) / (xf - xi)
  β = x̃i - α * xi
  return α, β
end

"""
exp(α * x), x ∈ [0, 1)

xi, xf = 0.0, 1.0
n = length(s)
N = 2^n
h = (xf - xi) / N
x = [xi + h * j for j in 0:(N-1)]
"""
function qtt(::typeof(exp), α::Number, β::Number, s::Vector{<:Index})
  n = length(s)
  return exp(β) * MPS([itensor([1, exp(α * 2.0^(-j))], s[j]) for j in 1:n])
end

function qtt(::typeof(exp), α::Number, s::Vector{<:Index})
  return qtt(exp, α, 0.0, s)
end

"""
sin(α * x), x ∈ [0, 1)

n = 5
lineplot(abs.(mps_to_discrete_function(qtt(sin, 2π/2^n, siteinds("Qubit", n)))))
"""
function qtt(::typeof(sin), α::Number, β::Number, s::Vector{<:Index})
  ψ₁ = qtt(exp, im * α, β, s)
  ψ₂ = qtt(exp, -im * α, β, s)

  # Workaround for bug in MPS + MPS with missing links
  ψ₁ = insert_missing_links(ψ₁)
  ψ₂ = insert_missing_links(ψ₂)

  return -(ψ₁, ψ₂; alg="directsum") / 2im
end

function qtt(::typeof(sin), α::Number, s::Vector{<:Index})
  return qtt(sin, α, 0.0, s)
end

"""
sin(α * x), x ∈ [0, 1)
"""
function qtt(::typeof(cos), α::Number, β::Number, s::Vector{<:Index})
  ψ₁ = qtt(exp, im * α, β, s)
  ψ₂ = qtt(exp, -im * α, β, s)

  # Workaround for bug in MPS + MPS with missing links
  ψ₁ = insert_missing_links(ψ₁)
  ψ₂ = insert_missing_links(ψ₂)

  return +(ψ₁, ψ₂; alg="directsum") / 2
end

function qtt(::typeof(cos), α::Number, s::Vector{<:Index})
  return qtt(cos, α, 0.0, s)
end

"""
Compute the square Euclidean distance:
|Ax - b|² = <Ax - b, Ax - b>
          = <Ax, Ax> - <Ax, b> - <b, Ax> + <b, b>
          = <Ax, Ax> - 2 * real(<b, Ax>) + <b, b>
See: https://github.com/JuliaStats/Distances.jl
"""
function sqeuclidean((A, x)::Tuple{MPO,MPS}, b::MPS)
  return inner(A, x, A, x) - 2 * real(inner(b', A, x)) + inner(b, b)
end

function sqeuclidean_normalized((A, x)::Tuple{MPO,MPS}, b::MPS)
  bb = inner(b, b)
  return 1.0 + (inner(A, x, A, x) / bb - 2 * real(inner(b', A, x)) / bb)
end

"""
Compute the square Euclidean distance:
|x - y|² = <x - y, x - y>
          = <x, x> - <x, y> - <y, x> + <y, y>
          = <x, x> - 2 * real(<y, x>) + <y, y>
See: https://github.com/JuliaStats/Distances.jl
"""
function sqeuclidean(x::MPS, y::MPS)
  return inner(x, x) - 2 * real(inner(y, x)) + inner(y, y)
end

function sqeuclidean_normalized(x::MPS, y::MPS)
  yy = inner(y, y)
  return 1.0 + (inner(x, x) / yy - 2 * real(inner(y, x)) / yy)
end

"""
helmholtz_solver(0, 10) # Solve a single sin wave with 2^10 gridpoints
"""
function helmholtz_solver(nk, n; init=nothing, solver_kwargs=(;))
  solver_kwargs = (;
    nsweeps=40,
    cutoff=1e-15,
    outputlevel=0,
    nsite=2,
    solver_kwargs...
  )
  xi = 0.0
  xf = 1.0

  if isnothing(init)
    s = siteinds("Qubit", n)
    init = randomMPS(s)
  else
    s = siteinds(init)
  end

  N = 2^n

  # Wavenumber
  h = (xf - xi) / 2^n
  k = 2π * 2^nk

  # Eigenvalue renormalized by the stepsize
  # λ = -(k * h)^2
  λ = -(2π * (xf - xi) * 2.0^(nk - n))^2

  # Set effective stepsize to 1
  A = laplacian_mpo(s, 1.0)

  @show norm(A)

  @show solver_kwargs

  solve_time = @elapsed begin
    u = dmrg_target(A, init; target_eigenvalue=λ, solver_kwargs...)
  end

  @show solve_time

  @show nk, n
  @show maxlinkdim(u)
  λ̃ = inner(u', A, u) / inner(u, u)
  @show λ̃, λ
  @show abs(λ - λ̃)
  @show abs(λ - λ̃) / λ

  # Compute the error ∫dx |Au(x) - (-k²u(x))|² / ∫dx |-k²u(x)|² = ∑ᵢ |Auᵢ + k²uᵢ| / ∑ᵢ |-k²uᵢ|
  @show sqeuclidean_normalized((A, u), λ * u)

  @show (inner(A, u, A, u) - inner(u', A, u)^2) / inner(u, u)

  return (; u, init, xi, xf, nk, solve_time, solver_kwargs)
end

helmholtz_solver_filename(nk, n) = "helmholtz_solver_nk_$(nk)_n_$(n).jld2"

"""
nks = 0:20
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/helmholtz_solver"
results_dir = joinpath(root_dir, "results")
helmholtz_solver_run(nks, Dict([nk => (nk + 2):(nk + 13) for nk in nks]); results_dir)
"""
function helmholtz_solver_run(nks, ns; results_dir, multigrid=true, solver_kwargs=(;))
  if !isdir(results_dir)
    @warn "Making the directory path $(results_dir)"
    mkpath(results_dir)
  end
  for nk in nks
    @show nk
    for n in ns[nk]
      @show n
      init = nothing
      if multigrid
        println("\nTry using state from length $(n - 1) as starting state.")
        init_filename = joinpath(results_dir, helmholtz_solver_filename(nk, n - 1))
        if !isfile(init_filename)
          @warn "File $init_filename doesn't exist, a random starting state will be used instead."
        else
          init = load(init_filename, "results").u
          init = prolongate(init, siteind("Qubit", n))
        end
      end
      results = helmholtz_solver(nk, n; init, solver_kwargs)
      println("Saving to $(results_dir)")
      jldsave(joinpath(results_dir, helmholtz_solver_filename(nk, n)); results)
    end
  end
end

"""
nks = 0:5:20
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/helmholtz_solver"
results_dir = joinpath(root_dir, "results")
plots_dir = joinpath(root_dir, "plots")
errors, solve_times, maxlinkdims = helmholtz_solver_analyze(nks, Dict([nk => (nk + 2):(nk + 13) for nk in nks]); results_dir, plots_dir);
"""
function helmholtz_solver_analyze(nks, ns; results_dir, plots_dir)
  if !isdir(results_dir)
    error("No results directory $(results_dir)")
  end
  if !isdir(plots_dir)
    @warn "Making the directory path $(plots_dir)"
    mkpath(plots_dir)
  end
  error_variances = Dict()
  error_eigvals = Dict()
  error_diffs = Dict()
  solve_times = Dict()
  maxlinkdims = Dict()
  for nk in nks
    @show nk
    error_variances[nk] = Float64[]
    error_eigvals[nk] = Float64[]
    error_diffs[nk] = Float64[]
    solve_times[nk] = Float64[]
    maxlinkdims[nk] = Float64[]
    for n in ns[nk]
      println()
      @show nk, n

      results = load(joinpath(results_dir, helmholtz_solver_filename(nk, n)), "results")
      u = results.u
      normalize!(u)

      # Exact norm is:
      # 2^((n - 1)/ 2)
      @show norm(u)

      u /= √2
      u .*= √2

      @show norm(u)

      xi = results.xi
      xf = results.xf
      solve_time = results.solve_time

      s = siteinds(u)
      n = length(u)
      N = 2^n
      h = (xf - xi) / N
      A = laplacian_mpo(s, 1.0)

      # Wavenumber
      k = 2 * π * 2^nk
      # λ = -(k * h)^2 = -((2π * 2^nk) * ((xf - xi) / 2^n))^2
      λ = -(2π * (xf - xi) * 2.0^(nk - n))^2

      λ̃ = inner(u', A, u) / inner(u, u)

      @show λ, λ̃, abs(λ - λ̃) / abs(λ)
      # error_eigval = @show abs(1 - λ̃ / λ)
      error_eigval = @show abs(λ - λ̃)

      # error_eig = @show abs(sqeuclidean_normalized((A, u), λ̃ * u))
      @show inner(A, u, A, u)
      @show inner(u', A, u)^2
      @show inner(u, u)

      # error_variance = @show abs((inner(A, u, A, u) / uu) - λ̃^2) / abs(λ̃)
      ũ = u / norm(u)
      error_variance = @show abs(inner(A, ũ, A, ũ) - inner(ũ', A, ũ)^2)
      if iszero(error_variance)
        error_variance = 1e-16
      end

      # QTT function range
      xi, xf = 0, 1 - 2.0^(-n)

      # Error is uniform
      x̃i, x̃f = 2.0^(-n), 1 - 2.0^(-n)

      # Middle is more accurate
      # x̃i, x̃f = 0, 1

      # Right side is more accurate
      # x̃i, x̃f = 0, 1 - 2.0^(-n)

      # Right side is more accurate
      # x̃i, x̃f = 2.0^(-n), 1

      α, β = rescale(xi, xf, x̃i, x̃f)
      u_exact = qtt(sin, α * k, β, s)
      @show nk, n
      @show 2^((n - 1)/ 2)

      @show sign(inner(u, u_exact))
      u *= real(sign(inner(u, u_exact)))

      @show norm(u_exact), norm(u)

      # @show display(lineplot(real.(mps_to_discrete_function(u_exact)); title="Exact"))
      # @show display(lineplot(mps_to_discrete_function(u); title="QTT"))
      # @show display(lineplot(mps_to_discrete_function(u) - real.(mps_to_discrete_function(u_exact)); title="Diff"))

      error_diff = @show sqeuclidean_normalized(u, u_exact)

      push!(error_variances[nk], error_variance)
      push!(error_eigvals[nk], error_eigval)
      push!(error_diffs[nk], real.(error_diff))
      push!(solve_times[nk], solve_time)
      push!(maxlinkdims[nk], maxlinkdim(u))
    end
  end

  # Make plots
  plot_error_variances = plot(;
    title="Helmholtz QTT solver",
    legend=:topright,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|Auᵢ|² - |∑ᵢ uⱼAuᵢ|²",
  )
  plot_error_eigvals = plot(;
    title="Helmholtz QTT solver",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="|λ - λ̃|",
  )
  plot_error_diffs = plot(;
    title="Helmholtz QTT solver",
    legend=:bottomleft,
    xaxis=:log,
    yaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="∑ᵢ|uᵢ - ũᵢ|² / ∑ᵢ|ũᵢ|²",
  )
  plot_maxlinkdims = plot(;
    title="Helmholtz QTT solver",
    legend=:topleft,
    xaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="QTT Rank",
  )
  plot_time = plot(;
    title="Helmholtz QTT solver",
    legend=:topleft,
    xaxis=:log,
    linewidth=3,
    xlabel="Number of gridpoints",
    ylabel="Time to solver (seconds)",
  )
  for nk in nks
    plot!(plot_error_variances, 2 .^ ns[nk], abs.(error_variances[nk]);
      label="k=2π * 10^$(round(nk * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_error_eigvals, 2 .^ ns[nk], abs.(error_eigvals[nk]);
      label="k=2π * 10^$(round(nk * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_error_diffs, 2 .^ ns[nk], abs.(error_diffs[nk]);
      label="k=2π * 10^$(round(nk * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_maxlinkdims, 2 .^ ns[nk], maxlinkdims[nk];
      label="k=2π * 10^$(round(nk * log10(2); digits=2))",
      linewidth=3,
    )
    plot!(plot_time, 2 .^ ns[nk], solve_times[nk];
      label="k=2π * 10^$(round(nk * log10(2); digits=2))",
      linewidth=3,
    )
  end

  println("Saving plots to $(plots_dir)")
  Plots.savefig(plot_error_variances, joinpath(plots_dir, "plot_error_variances.png"))
  Plots.savefig(plot_error_eigvals, joinpath(plots_dir, "plot_error_eigvals.png"))
  Plots.savefig(plot_error_diffs, joinpath(plots_dir, "plot_error_diffs.png"))
  Plots.savefig(plot_maxlinkdims, joinpath(plots_dir, "plot_qtt_rank.png"))
  Plots.savefig(plot_time, joinpath(plots_dir, "plot_solve_time.png"))
  return errors, solve_times, maxlinkdims
end

"""
nk = 18
n = 32 # 28:34
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/helmholtz_solver"
results_dir = joinpath(root_dir, "results")
helmholtz_solver_visualize_solution(nk, n, 0.75, -1; results_dir, plots_dir)
"""
function helmholtz_solver_visualize_solution(; nk::Int, n::Int, xstart::Float64, zoom::Int=0, kwargs...)
  # nzoom is the zoom level relative to the length scale `nk`
  nproj = nk + zoom

  # Assume range [xi, xf) = [0.0, 1.0)

  if xstart ≥ 1
    # Catch overflow
    proj = fill(1, n)
  else
    proj = reverse(digits(Int(round(xstart * 2^n)); base=2, pad=n))
  end
  proj = proj[1:nproj]
  return helmholtz_solver_visualize_solution(nk, n, proj; kwargs...)
end

function helmholtz_solver_visualize_solution(nk::Int, n::Int, proj::Vector{Int}; results_dir, nplot=min(8, n))
  nplot = min(n, nplot)

  if nplot + length(proj) > n
    nplot = n - length(proj)
  end

  nproj = length(proj)
  xstart = sum([proj[j] * 2.0^(-j) for j in 1:nproj])
  xstop = xstart + sum([2.0^(-j) for j in (nproj + 1):n])
  xrange = range(; start=xstart, stop=xstop, length=2^nplot)

  results = load(joinpath(results_dir, helmholtz_solver_filename(nk, n)), "results")
  u = results.u
  s = siteinds(u)
  normalize!(u)
  u /= √2
  u .*= √2

  k = 2 * π * 2^nk

  # QTT function range
  xi, xf = 0, 1 - 2.0^(-n)

  # Error is uniform
  x̃i, x̃f = 2.0^(-n), 1 - 2.0^(-n)

  # Middle is more accurate
  # x̃i, x̃f = 0, 1

  # Right side is more accurate
  # x̃i, x̃f = 0, 1 - 2.0^(-n)

  # Right side is more accurate
  # x̃i, x̃f = 2.0^(-n), 1

  α, β = rescale(xi, xf, x̃i, x̃f)

  u_exact = qtt(sin, α * k, β, s)

  u *= real(sign(inner(u, u_exact)))

  # @show sqeuclidean_normalized(u, u_exact)
  # @show sqeuclidean(u, u_exact)

  nleft = length(proj)
  nright = n - nplot - nleft

  # @show nplot, nleft, nright

  left_bits = proj

  # @show norm(u)
  # @show norm(u_exact)

  u_plot = project_bits(u, left_bits, nright)
  u_exact_plot = project_bits(u_exact, left_bits, nright)

  u_vec = mps_to_discrete_function(u_plot)
  u_exact_vec = real.(mps_to_discrete_function(u_exact_plot))

  return (; xrange, u_vec, u_exact_vec)
end

"""
nks = 18:18
ns = [30, 32, 34] # 28:34
xstarts = 0:0.5:1.0
zoom = -2
root_dir = "$(ENV["HOME"])/workdir/ITensorPartialDiffEq.jl/helmholtz_solver"
results_dir = joinpath(root_dir, "results")
plots_dir = joinpath(root_dir, "plots")
helmholtz_solver_plot_solutions(nks, ns, xstarts, zoom; results_dir, plots_dir)
"""
function helmholtz_solver_plot_solutions(nks, ns, xstarts, zoom; results_dir, plots_dir)
  for nk in nks
    for xstart in xstarts
      plot_u_diff = plot(;
        title="Helmholtz solution error, k = 2π * 2^$(nk)",
        legend=:bottomright,
        linewidth=3,
        xlabel="x",
        ylabel="|u(x) - sin(kx)|",
        yaxis=:log,
        xformatter=:plain, # disable scientific notation
        yrange=[1e-8, 1e-2],
      )
      for n in ns
        (; xrange, u_vec, u_exact_vec) = helmholtz_solver_visualize_solution(; nk, n, xstart, zoom, results_dir)

        # Plot the original functions
        plot_u = plot(;
                      title="Helmholtz solution, k = 2π * 2^$(nk), n = 2^$(n)",
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
          label="sin(kx)",
          linewidth=3,
        )
        Plots.savefig(plot_u, joinpath(plots_dir, "helmholtz_visualize_nk_$(nk)_n_$(n)_xstart_$(xstart)_zoom_$(zoom)_qtt.png"))

        # Plot the error
        plot!(plot_u_diff, xrange, abs.(u_vec - u_exact_vec);
          label="2^$n gridpoints",
          linewidth=3,
        )
      end
      Plots.savefig(plot_u_diff, joinpath(plots_dir, "helmholtz_visualize_nk_$(nk)_ns_$(ns)_xstart_$(xstart)_zoom_$(zoom)_diff.png"))
    end
  end
end
