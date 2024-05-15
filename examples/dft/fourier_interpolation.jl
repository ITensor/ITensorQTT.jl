using ITensors
using ITensorMPS
using ITensorQTT
using FFTW
using UnicodePlots

n = 5
s = siteinds("Qubit", n)

x₀ = 0.5
σ = 0.1
f(x) = exp(-(x - x₀)^2 / σ^2)
ψ = function_to_mps(f, s, 0.0, 1.0; cutoff=1e-15)
ℱψ = apply_dft_mpo(ψ; cutoff=1e-15)
ℱ⁻¹ℱψ = apply_idft_mpo(ℱψ; cutoff=1e-15)

display(lineplot(mps_to_discrete_function(ψ); title="ψ"))
display(lineplot(real(mps_to_discrete_function(ℱψ)); title="real(ℱψ)"))
display(lineplot(imag(mps_to_discrete_function(ℱψ)); title="real(ℱψ)"))
display(lineplot(real(mps_to_discrete_function(ℱ⁻¹ℱψ)); title="real(ℱ⁻¹ℱψ)"))

@show norm(fft(mps_to_discrete_function(ψ)) / 2^(n / 2) - mps_to_discrete_function(ℱψ))
@show norm(ℱ⁻¹ℱψ - ψ)

k = 3
ψⁿ⁺ᵏ = fourier_interpolation(ψ, k; cutoff=1e-15)
sⁿ⁺ᵏ = siteinds(ψⁿ⁺ᵏ)
ψ̃ⁿ⁺ᵏ = function_to_mps(f, sⁿ⁺ᵏ, 0.0, 1.0; cutoff=1e-15)

display(lineplot(real(mps_to_discrete_function(ψⁿ⁺ᵏ)); title="real(ψⁿ⁺ᵏ)"))
@show norm(ψⁿ⁺ᵏ - ψ̃ⁿ⁺ᵏ)
