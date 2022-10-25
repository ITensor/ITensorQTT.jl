
# Airy equation compression

Compress the Airy equation solution using a QTT/MPS on an interval `xi = 1.0` to `xf` using a grid of size `N = 2^n`.

Collect results with:
```julia
include("airy_equation.jl")
nxfs = 1:20 # xf in [2^1, 2^2, ..., 2^20]
ns = 1:5 # n in [2^1, 2^2, ..., 2^22]
α, β = 1.0, 1.0 # Boundary conditions `u(xi) = α Ai(-xi) + β Bi(-xi)`, `u(xf) = α Ai(-xf) + β Bi(-xf)`
root_dir = "$(ENV["HOME"])/workdir/ITensorQTT.jl/airy_solution_compression"
results_dir = joinpath(root_dir, "results")
cutoff = 1e-15 # QTT/MPS compression cutoff
airy_qtt_compression_get_results(nxfs, ns; α, β, results_dir, cutoff)
```

Plot the results with:
```julia
include("airy_equation.jl")
nxfs = 1:2:11
ns = Dict(1 => 6:22, 3 => 6:22, 5 => 8:22, 7 => 11:22, 9 => 14:22, 11 => 17:22)
best_fit_points = Dict(1 => 6:16, 3 => 6:18, 5 => 8:20, 7 => 12:21, 9 => 18:22, 11 => 20:22)
root_dir = "$(ENV["HOME"])/workdir/ITensorQTT.jl/airy_solution_compression"
results_dir = joinpath(root_dir, "results")
plots_dir = joinpath(root_dir, "results")
airy_qtt_compression_plot_results(nxfs, ns; results_dir, plots_dir, best_fit_points)
```

# Airy equation QTT/MPS solver

Run the QTT Airy equation solver in the interval `xⁱ = 1.0` to `xᶠ` on grids of size `N = 2^n`:
```julia
include("airy_equation.jl")
run_airy(; xᶠ=2.0, n=10:11, multigrid=true, save_results=true, dirname="results_multigrid", linsolve_kwargs=(; nsweeps=20))
```
Results will be saved in the directory `dirname` if `save_results=true`.

Compare to exact results and create plots as follows:
```julia
include("airy_equation_qtt_solver_plot.jl")
```
Modify the script by changing the lines:
```julia
xᶠs = [2.0, 30.0]
nss = Dict(2.0 => 10:20, 30.0 => 10:20)
```
to choose which final interval values and which grid discretizations to plot.
