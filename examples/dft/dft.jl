using ITensors
using ITensorMPS
using ITensorQTT
using FFTW
using UnicodePlots

ITensors.disable_warn_order()

function dft_matrix(n::Int)
  N = 2^n
  ω = exp(-2π * im / N)
  return [ω^(j * k) / √N for j in 0:(N-1), k in 0:(N-1)]
end

n = 20
s = siteinds("Qubit", n)

println("\nMake DFT MPO")
U = @time dft_mpo(s)
@show maxlinkdim(U)

@show norm(U - U_gates)

if n ≤ 10
  println("\nMake DFT Matrix")
  dft_mat = @time dft_matrix(n)

  println("\nConvert DFT MPO to Matrix")
  U_mat = @time mpo_to_mat(U; reverse_output_sites=true)

  println("\nCompare DFT MPO to DFT Matrix")
  @show norm(U_mat - dft_mat)
end

ψ = +(qtt(sin, 2π, s), qtt(sin, 4π, s), qtt(sin, 8π, s), qtt(sin, 16π, s); alg="directsum")
@show maxlinkdim(ψ)
  
println("\nApply DFT MPO")
ψ̃ = @time reverse(apply(U, ψ; cutoff=1e-15))
@show maxlinkdim(ψ̃)

println("\nConvert MPS to Vector")
f = @time mps_to_discrete_function(ψ)

println("\nConvert Fourier transformed MPS to Vector")
f̃ = @time mps_to_discrete_function(ψ̃)

println("\nFFT")
f̃_fft = @time fft(f) / √(2^n)

@show norm(f̃_fft - f̃) / norm(f̃)
