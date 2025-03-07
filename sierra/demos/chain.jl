# Basic rules

using LinearAlgebra

A, b = rand(2, 3), rand(2)
residuals(x) = A * x - b
∂(::typeof(residuals)) = x -> (u -> A * u)  # ℝ³ → ℝ²
∂ᵀ(::typeof(residuals)) = x -> (v -> adjoint(A) * v)  # ℝ² → ℝ³ 

sqnorm(r) = sum(abs2, r)
∂(::typeof(sqnorm)) = r -> (v -> dot(2r, v))  # ℝ² → ℝ
∂ᵀ(::typeof(sqnorm)) = r -> (w -> 2r .* w)  # ℝ → ℝ²

# Composition

function ∂(f::ComposedFunction)
    g, h = f.outer, f.inner
    return x -> ∂(g)(h(x)) ∘ ∂(h)(x)
end

function ∂ᵀ(f::ComposedFunction)
    g, h = f.outer, f.inner
    return x -> ∂ᵀ(h)(x) ∘ ∂ᵀ(g)(h(x))
end

import ForwardDiff as FD, Zygote

f = sqnorm ∘ residuals;
x, Δx = rand(3), [1, 0, 0];

∂(f)(x)(Δx)  # partial derivative
∂ᵀ(f)(x)(1)  # gradient

FD.derivative(t -> f(x + t * Δx), 0)
Zygote.gradient(f, x)[1]
