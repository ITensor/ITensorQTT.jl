using ITensorQTT
using ITensors
using HCubature
using Test
using FFTW

@testset "ITensorQTT.jl" begin
  @testset "MPS vcat" begin
    n = 4
    s = siteinds("S=1/2", n)
    ψ = randomMPS(s[1:(end - 1)])
    A = randomITensor(s[end])
    @test [A; ψ] isa MPS
    @test [ψ; A] isa MPS
    @test [MPS([A]); ψ] isa MPS
    @test [ψ; MPS([A])] isa MPS
    @test length([A; ψ]) == n
    @test length([ψ; A]) == n
    @test length([MPS([A]); ψ]) == n
    @test length([ψ; MPS([A])]) == n
    @test length(reverse(ψ)) == n - 1
    @test reverse(ψ)[1] ≈ ψ[end]
  end

  @testset "MPS interleave" begin
    n = 3
    s = siteinds("S=1/2", n)
    ψ = randomMPS(s)
    ϕ = randomMPS(s)
    ρ = ITensorQTT.interleave(ψ, ϕ)
    @test length(ρ) == 2n
    @test contract(ψ) * contract(ϕ) ≈ contract(ρ)
  end

  @testset "Function to MPS conversion: 1-dimension" begin
    f(x) = sin(π * x)
    nbits = 8
    s = siteinds("Qubit", nbits)
    xstart, xstop = 0.0, 1.0
    x = range(; start=xstart, stop=xstop, length=(2^nbits + 1))[1:(end - 1)]

    # Defaults to `alg="factorize"`
    ψ = function_to_mps(f, s, xstart, xstop)
    @test length(ψ) == nbits
    @test maxlinkdim(ψ) == 2
    f̃ = mps_to_discrete_function(ψ)
    @test f̃ ≈ f.(x)

    ψ = function_to_mps(f, s, xstart, xstop; alg="factorize", cutoff=1e-15)
    @test length(ψ) == nbits
    @test maxlinkdim(ψ) == 2
    f̃ = mps_to_discrete_function(ψ)
    @test f̃ ≈ f.(x)

    ψ = function_to_mps(f, s, xstart, xstop; alg="factorize", cutoff=1e-8)
    @test length(ψ) == nbits
    @test maxlinkdim(ψ) == 2
    f̃ = mps_to_discrete_function(ψ)
    @test f̃ ≈ f.(x)

    ψ = function_to_mps(f, s, xstart, xstop; alg="polynomial", degree=8, cutoff=1e-8)
    @test length(ψ) == nbits
    @test maxlinkdim(ψ) == 2
    f̃ = mps_to_discrete_function(ψ)
    @test f̃ ≈ f.(x) rtol = 1e-6

    ψ = function_to_mps(
      f, s, xstart, xstop; alg="polynomial", degree=8, length=50, cutoff=1e-8
    )
    @test length(ψ) == nbits
    @test maxlinkdim(ψ) == 2
    f̃ = mps_to_discrete_function(ψ)
    @test f̃ ≈ f.(x) rtol = 1e-6

    ψ = function_to_mps(f, s, xstart, xstop; alg="recursive", cutoff=1e-15)
    @test length(ψ) == nbits
    @test maxlinkdim(ψ) == 2
    f̃ = mps_to_discrete_function(ψ)
    @test f̃ ≈ f.(x)
  end

  @testset "Function to MPS conversion: 2-dimension" begin
    ndims = 2
    f₁(x₁) = sin(2π * x₁)
    f₂(x₂) = sin(π * x₂)
    nbits₁ = 4
    nbits₂ = 4
    s₁ = siteinds("Qubit", nbits₁)
    s₂ = siteinds("Qubit", nbits₂)
    x₁start, x₁stop = 0.0, 1.0
    x₂start, x₂stop = 0.0, 1.0
    x₁ = range(; start=x₁start, stop=x₁stop, length=(2^nbits₁ + 1))[1:(end - 1)]
    x₂ = range(; start=x₂start, stop=x₂stop, length=(2^nbits₂ + 1))[1:(end - 1)]

    f = (f₁, f₂)
    F(x) = prod(j -> f[j](x[j]), 1:ndims)
    s = (s₁, s₂)
    xstart = (x₁start, x₂start)
    xstop = (x₁stop, x₂stop)
    nbits = (nbits₁, nbits₂)
    x = [(x₁ᵢ, x₂ⱼ) for x₁ᵢ in x₁, x₂ⱼ in x₂]

    # Defaults to `alg="factorize"`
    ψ = function_to_mps(f, s, xstart, xstop)
    @test length(ψ) == sum(nbits)
    @test maxlinkdim(ψ) == 4
    f̃ = mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ map(xᵢ -> F(xᵢ), x)

    ψ = function_to_mps(f, s, xstart, xstop; alg="factorize", cutoff=1e-15)
    @test length(ψ) == sum(nbits)
    @test maxlinkdim(ψ) == 4
    f̃ = mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ map(xᵢ -> F(xᵢ), x)

    ψ = function_to_mps(f, s, xstart, xstop; alg="factorize", cutoff=1e-8)
    @test length(ψ) == sum(nbits)
    @test maxlinkdim(ψ) == 4
    f̃ = mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ map(xᵢ -> F(xᵢ), x)

    ψ = function_to_mps(f, s, xstart, xstop; alg="polynomial", degree=8, cutoff=1e-8)
    @test length(ψ) == sum(nbits)
    @test maxlinkdim(ψ) == 4
    f̃ = mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ map(xᵢ -> F(xᵢ), x) rtol = 1e-4

    ψ = function_to_mps(
      f, s, xstart, xstop; alg="polynomial", degree=8, length=50, cutoff=1e-8
    )
    @test length(ψ) == sum(nbits)
    @test maxlinkdim(ψ) == 6
    f̃ = mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ map(xᵢ -> F(xᵢ), x) rtol = 1e-3

    ψ = function_to_mps(f, s, xstart, xstop; alg="recursive", cutoff=1e-15)
    @test length(ψ) == sum(nbits)
    @test maxlinkdim(ψ) == 4
    f̃ = mps_to_discrete_function(Val(2), ψ)
    @test f̃ ≈ map(xᵢ -> F(xᵢ), x)
  end

  @testset "Laplacian MPO" begin
    nbits = 3

    # xstep = 1.0
    A = ITensorQTT.laplacian_matrix(2^nbits)
    s = siteinds("Qubit", nbits)
    M = laplacian_mpo(s)
    @test ITensorQTT.mpo_to_mat(M) ≈ A

    xstep = 1 / 2^nbits
    A = ITensorQTT.laplacian_matrix(2^nbits, xstep)
    s = siteinds("Qubit", nbits)
    M = laplacian_mpo(s, xstep)
    @test ITensorQTT.mpo_to_mat(M) ≈ A
  end

  @testset "Function integration - 1-dimensional" begin
    f(x) = x^2 * sin(π * x)
    xstart = 0.0
    xstop = 1.0
    int_exact = hquadrature(f, xstart, xstop)[1]
    nbits = 12
    s = siteinds("Qubit", nbits)
    int_mps = integrate_mps(function_to_mps(f, s, xstart, xstop))
    @test int_exact ≈ int_mps rtol = 1e-7
  end

  @testset "Function integration - 2-dimensional" begin
    fˣ(x) = x^2 * sin(π * x)
    fʸ(y) = y * sin(π * y) + y^2
    f = (fˣ, fʸ)
    xstart = (0.0, 0.0)
    xstop = (1.0, 1.0)
    int_exact = hcubature(x -> f[1](x[1]) * f[2](x[2]), xstart, xstop)[1]
    nbits = 12
    s = (siteinds("Qubit", nbits), siteinds("Qubit", nbits))
    int_mps = integrate_mps(function_to_mps(f, s, xstart, xstop))
    @test int_exact ≈ int_mps rtol = 1e-3
  end

  @testset "Prolongation and retraction - 1-dimensional" begin
    f(x) = sin(π * x)
    xstart = 0.0
    xstop = 1.0
    nbits = 12
    s = siteinds("Qubit", nbits)

    # MPS at different scales
    ψ₀ = function_to_mps(f, s[1:(end - 2)], xstart, xstop)
    ψ₁ = function_to_mps(f, s[1:(end - 1)], xstart, xstop)
    ψ₂ = function_to_mps(f, s, xstart, xstop)

    # Prolongate by one
    ψ₂′ = prolongate(ψ₁, s[end])
    @test ψ₂′ ≈ prolongate(ψ₁, (s[end],))
    @test ψ₂′ ≈ prolongate(ψ₁, [s[end]])
    @test ψ₂′ ≈ prolongate(ψ₁, ([s[end]],))
    @test ψ₂′ ≈ ψ₂ rtol = 1e-6

    # Prolongate by two
    ψ₂′ = prolongate(ψ₀, [s[end - 1], s[end]])
    @test ψ₂′ ≈ ψ₂ rtol = 1e-6

    # Retract by one
    ψ₁′ = retract(ψ₂)
    @test ψ₁′ ≈ retract(ψ₂, 1)
    @test ψ₁′ ≈ ψ₁ rtol = 1e-5

    # Retract by two
    ψ₀′ = retract(ψ₂, 2)
    @test ψ₀′ ≈ ψ₀ rtol = 1e-5
  end

  @testset "Prolongation and retraction - 2-dimensional" begin
    fˣ(x) = sin(π * x)
    fʸ(y) = sin(π * y)
    f = (fˣ, fʸ)
    xstart = (0.0, 0.0)
    xstop = (1.0, 1.0)
    nbits = 10
    s = (siteinds("Qubit", nbits), siteinds("Qubit", nbits))

    # MPS at different scales
    ψ₀ = function_to_mps(f, map(sⱼ -> sⱼ[1:(nbits - 2)], s), xstart, xstop)
    ψ₁ = function_to_mps(f, map(sⱼ -> sⱼ[1:(nbits - 1)], s), xstart, xstop)
    ψ₂ = function_to_mps(f, s, xstart, xstop)

    # Prolongate by one
    ψ₂′ = prolongate(ψ₁, map(sⱼ -> sⱼ[end], s))
    @test ψ₂′ ≈ prolongate(ψ₁, map(sⱼ -> [sⱼ[end]], s))
    @test ψ₂′ ≈ ψ₂ rtol = 1e-4

    # Prolongate by two
    ψ₂′ = prolongate(ψ₀, map(sⱼ -> [sⱼ[end - 1], sⱼ[end]], s))
    @test ψ₂′ ≈ ψ₂ rtol = 1e-3

    # Retract by one
    ψ₁′ = retract(ψ₂, (1, 1))
    @test ψ₁′ ≈ ψ₁ rtol = 1e-4

    # Retract by two
    ψ₀′ = retract(ψ₂, (2, 2))
    @test ψ₀′ ≈ ψ₀ rtol = 1e-3
  end

  @testset "QFT/DFT/FFT" begin
    n = 5
    s = siteinds("Qubit", n)
    ℱ = dft_mpo(s)
    @test maxlinkdim(ℱ) < 10
    @test mpo_to_mat(ℱ) ≈ ITensorQTT.dft_matrix(n)
    ψ = +(
      qtt(sin, 2π, s), qtt(sin, 4π, s), qtt(sin, 8π, s), qtt(sin, 16π, s); alg="directsum"
    )

    # Perform QTT-DFT
    ℱψ = reverse(apply(ℱ, ψ; cutoff=1e-15))
    f = mps_to_discrete_function(ψ)
    ℱf = mps_to_discrete_function(ℱψ)
    ℱf_fft = fft(f) / 2^(n / 2)
    @test ℱf_fft ≈ ℱf
    @test apply_idft_mpo(ℱψ) ≈ ψ

    # Perform QTT-DFT
    ℱψ = apply_dft_mpo(ψ; cutoff=1e-15)
    f = mps_to_discrete_function(ψ)
    ℱf = mps_to_discrete_function(ℱψ)
    ℱf_fft = fft(f) / 2^(n / 2)
    @test ℱf_fft ≈ ℱf
    @test apply_idft_mpo(ℱψ) ≈ ψ
  end

  @testset "Fourier interpolation" begin
    n = 5
    s = siteinds("Qubit", n)
    x₀ = 0.5
    σ = 0.1
    f(x) = exp(-(x - x₀)^2 / σ^2)
    ψ = function_to_mps(f, s, 0.0, 1.0; cutoff=1e-15)

    # Test DFT
    ℱψ = apply_dft_mpo(ψ)
    ℱ⁻¹ℱψ = apply_idft_mpo(ℱψ)
    @test fft(mps_to_discrete_function(ψ)) / 2^(n / 2) ≈ mps_to_discrete_function(ℱψ)
    @test ℱ⁻¹ℱψ ≈ ψ

    k = 2
    ψⁿ⁺ᵏ = fourier_interpolation(ψ, k)
    @test length(ψⁿ⁺ᵏ) == n + k
    sⁿ⁺ᵏ = siteinds(ψⁿ⁺ᵏ)
    ψ̃ⁿ⁺ᵏ = function_to_mps(f, sⁿ⁺ᵏ, 0.0, 1.0; cutoff=1e-15)
    @test ψⁿ⁺ᵏ ≈ ψ̃ⁿ⁺ᵏ
  end

  @testset "Repeat function" begin
    n = 3
    s = siteinds("Qubit", n)
    x₀ = 0.5
    σ = 0.1
    g(x) = exp(-(x - x₀)^2 / σ^2)
    ψ = function_to_mps(g, s, 0.0, 1.0; cutoff=1e-15)
    v = mps_to_discrete_function(ψ)
    for k in 1:3
      ψ_repeat = repeat_function(ψ, k)
      v_repeat = mps_to_discrete_function(ψ_repeat)
      @test v_repeat ≈ repeat(v, 2^k)
    end
  end
end
