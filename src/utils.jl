number_of_zeros(v) = count(j -> sign(v[j]) ≠ sign(v[j + 1]), 1:(length(v) - 1))

linreg(x, y) = [fill!(similar(x), 1);; x] \ y

"""
Find `α, β` such that `x̃ = α * x + β` given
`(xi, x̃i)` and `(xf, x̃f)`.
"""
function rescale(xi, xf, x̃i, x̃f)
  α = (x̃f - x̃i) / (xf - xi)
  β = x̃i - α * xi
  return α, β
end
