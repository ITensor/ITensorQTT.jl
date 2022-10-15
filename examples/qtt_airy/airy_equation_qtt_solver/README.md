Run the QTT Airy equation solver in the interval `xⁱ = 1.0` to `xᶠ` on grids of size `N = 2^n`:
```julia
include("airy_equation_qtt_solver_run.jl")
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
