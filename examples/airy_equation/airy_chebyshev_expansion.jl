using FastChebInterp
using LinearAlgebra
using SpecialFunctions

# airy_solution(x, α=1.0, β=0.0) = α * airyai(-x) + β * airybi(-x)

# f(x) = sin(2x + 3cos(4x))
# x = chebpoints(200, 0, 10)
# c = chebinterp(f.(x), 0, 10)
function airy_chebyshev_expansion(xf, N, order; α=1.0, β=1.0, xi=1.0)
  α, β = (α, β) ./ norm((α, β))

  # N = 2^n
  # xf = 2^nxf
  # order = 2^norder

  h = (xf - xi) / N
  # xs = range(; start=xi, stop=xf, length=(N+1))[1:N]
  xs = [xi + j * h for j in 0:(N - 1)]

  println("Compute discrete Airy function")
  # u_exact = @time airy_solution.(xs, α, β)

  f(x) = 1/√2 * airyai(-x) + 1/√2 * airybi(-x)
  x = chebpoints(order, xi, xf)
  c = chebinterp(f.(x), xi, xf)

  fs = f.(xs)
  cs = c.(xs)
  return sum(abs2, cs - fs) / sum(abs2, fs)
end

"""
xfs = 100:100:500
orders = 100:100:5000
N = 2^20
optimal_orders = airy_chebyshev_scaling(xfs, N, orders)
# optimal_orders = [400, 1100, 2100, 3100, 4400]

a, b = linreg(log2.(xfs), log2.(optimal_orders))
# (a, b) = (-1.2647805229102491, 1.4905125067205558)

using UnicodePlots
p = lineplot(xfs, optimal_orders)
lineplot!(p, xfs, 2^a * (xfs .^ b))
norm(optimal_orders - 2^a * (xfs .^ b)) / norm(optimal_orders)
# 0.01230967827202544
"""
function airy_chebyshev_scaling(xfs, N, orders)
  optimal_orders = Int[]
  for xf in xfs
    for order in orders
      err = airy_chebyshev_expansion(xf, N, order)
      @show xf, N, order, err
      if err < 1e-3
        println("Error threshold of 1e-3 reached, save and break")
        push!(optimal_orders, order)
        break
      end
    end
  end
  return optimal_orders
end

"""
nxfs = 1:10
n = 20
optimal_norders = [1, 3, 4, 5, 7, 8, 10, 11, 13, 14]
a, b = linreg(nxfs, optimal_norders)
# (-0.333333333333332, 1.4424242424242424)
"""

"""
xfs = 2:10:102
N = 2^20
optimal_orders = [11, 21, 51, 81, 111, 151, 201, 251, 301, 351, 411]
a, b = linreg(log2.(xfs[7:end]), log2.(optimal_orders[7:end]))
# (-0.8199522400929715, 1.4235362413641646)
"""
