using JLD2

include("airy_equation_qtt.jl")

function run_airy(;
  xᶠ,
  n,
  dirname,
  save_results,
  multigrid, # Use results from shorter system as starting state
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
    linsolve_kwargs...,
  )
  @show linsolve_kwargs

  if !isdir(dirname)
    @warn "Results directory $dirname doesn't exist, making it now."
    mkdir(dirname)
  end

  xᶠs = xᶠ
  ns = n
  results = Matrix(undef, length(xᶠs), length(ns))
  for j in eachindex(xᶠ)
    xᶠⱼ = xᶠ[j]
    for k in eachindex(n)
      nₖ = n[k]
      println("\n" * "#"^100)
      println("Running Airy equation QTT solver for `xᶠ = $(xᶠⱼ)` and `n = $(nₖ)`.\n")
      u_init = nothing
      if multigrid
        println("\nTry using state from length $(nₖ - 1) as starting state.")
        init_filename = airy_filename(; dirname, xᶠ=xᶠⱼ, n=(nₖ - 1))
        if !isfile(init_filename)
          @warn "File $init_filename doesn't exist, a random starting state will be used instead."
        else
          u_init = load_airy_results(; dirname, xᶠ=xᶠⱼ, n=nₖ - 1).u
        end
      else
        @warn "Not saving results to file $filename."
      end
      result = solve_airy(; xᶠ=xᶠⱼ, n=nₖ, α, β, seed, init=u_init, linsolve_kwargs)
      results[j, k] = result
      filename = airy_filename(; dirname, xᶠ=xᶠⱼ, n=nₖ)
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
  return results
end
