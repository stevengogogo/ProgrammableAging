module OdeModel

    using ModelingToolkit

    export eqs, F_H, F_S


    @parameters t k₁ k₂ k₃ k₄ k₅ k₆ KM₁ KM₂ KM₃ KM₄ n₁ n₂ n₃ n₄ α β S_tot
    @variables H(t) S(t)
    @derivatives D'~t

    hill(s, km, n) = (s^n + km^n) /  (s^n + km^n)

    eqs = [
        D(H) ~ k₁ * (1-α)*hill(S, KM₁, n₁) * hill(H, KM₂, n₂) + k₂ - k₃*H,
        D(S) ~ k₄ * (1-β)*hill(H, KM₃,n₃) * hill(S, KM₄, n₄)*(S_tot - S) + k₅ - k₆*S
    ]

    # Flux
    F_H = eqs[1].rhs
    F_S = eqs[2].rhs

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
        S_tot => 225]

    u0 = [
        H => 100.0,
        S => 100.0
    ]


end
