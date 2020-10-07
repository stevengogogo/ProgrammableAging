using ModelingToolkit
using DifferentialEquations

@parameters t k₁ k₂ k₃ k₄ k₅ k₆ KM₁ KM₂ KM₃ KM₄ n₁ n₂ n₃ n₄ α β S
@variables H(t) S(t)
@derivatives D'~t


p =[k₁ => 40,
    k₂ => 1,
    k₃ => 0.004,
    k₄ => 0.02,
    k₅ => 0.04,
    k₆ => 0.004,
    KM₁ => 280,
    KM₂ => 4250,
    KM₃ => 2200,
    KM₄ => 90,
    n₁ => 2,
    n₂ => 4,
    n₃ => 2,
    n₄ => 4,
    α => 0.35,
    β => 0.95,
    S => 225]
