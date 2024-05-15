function boundary_value_mps(s::Vector{<:Index}, ui::Number, uf::Number)
    n = length(s)
    l = [Index(2; tags = "Link,l=$(j)↔$(j+1)") for j = 1:(n-1)]
    A = MPS(n)
    A[1] = itensor([1.0 0.0; 0.0 1.0], s[1], l[1])
    aⱼ = zeros(2, 2, 2)
    aⱼ[1, 1, 1] = 1.0
    aⱼ[2, 2, 2] = 1.0
    for j = 2:(n-1)
        A[j] = ITensor(aⱼ, l[j-1], s[j], l[j])
    end
    A[end] = itensor([ui 0.0; 0.0 uf], l[n-1], s[n])
    return A
end

function boundary_value_vector(s::Vector{<:Index}, ui::Number, uf::Number)
    n = length(s)
    N = 2^n
    b = zeros(N)
    b[1] = ui
    b[N] = uf
    return b
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
    return exp(β) * MPS([itensor([1, exp(α * 2.0^(-j))], s[j]) for j = 1:n])
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

    return -(ψ₁, ψ₂; alg = "directsum") / 2im
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

    return +(ψ₁, ψ₂; alg = "directsum") / 2
end

function qtt(::typeof(cos), α::Number, s::Vector{<:Index})
    return qtt(cos, α, 0.0, s)
end
