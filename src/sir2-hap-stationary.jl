
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

hill(s, km, n) = (s^n + km^n) /  (s^n + km^n)
norm2(X, a, b) = (X-a)^2.0 / b^2.0

@parameters H, S, θ
#@derivatives Dt'~t
@derivatives Dss''~S
@derivatives Dhh''~H
@derivatives Ds'~S
@derivatives Dh'~H
@variables p(..)

# Parameters
k₁ = 40.
k₂ = 1.
k₃ = 0.004
k₄ = 0.02
k₅ = 0.04
k₆ = 0.004
KM₁ = 280.
KM₂ = 4250.
KM₃ = 2200.
KM₄ = 90.
n₁ = 2.
n₂ = 4.
n₃ = 2.
n₄ = 4.
α = 0.35
β = 0.95
S_tot = 225.
D_H=500.
D_S=0.45

σ_H, σ_S = 5000.0, 150.0
H0, S0 = 3275.0, 112.0

# ODE
F_H = k₁ * (1. -α)*hill(S, KM₁, n₁) * hill(KM₁, S, n₁) * hill(H, KM₂, n₂) + k₂ - k₃*H
F_S = k₄ * (1. -β)*hill(H, KM₃,n₃)  * hill(KM₃, H, n₃)  * hill(S, KM₄, n₄)*(S_tot - S) + k₅ - k₆*S

##
using SymPy

S, H = symbols("S H")



F_H = k₁ * (1. -α)*hill(S, KM₁, n₁) * hill(KM₁, S, n₁) * hill(H, KM₂, n₂) + k₂ - k₃*H
F_S = k₄ * (1. -β)*hill(H, KM₃,n₃)  * hill(KM₃, H, n₃)  * hill(S, KM₄, n₄)*(S_tot - S) + k₅ - k₆*S # get the derivatives

## derivate here

# Discretization
dS = 1.0; dH=10.0;

# PDE
#eq =   -Dh(F_H)*p(S,H,θ)-F_H*Dh(p(S,H,θ)) - Ds(F_S)*p(S,H,θ) - F_S*Ds(p(S,H,θ)) + 500.0 * Dhh(p(S,H,θ)) + 0.45 *Dss(p(S,H,θ)) ~ 0
eq =   -(-0.004)*p(S,H,θ)-F_H*Dh(p(S,H,θ)) - (-0.005)*p(S,H,θ) - F_S*Ds(p(S,H,θ)) + 500.0 * Dhh(p(S,H,θ)) + 0.45 *Dss(p(S,H,θ)) ~ 0
      #1. ~ dS*dH*p(S,H,θ)


# Domains
domains = [ S ∈ IntervalDomain(0.0,112.0),
            H ∈ IntervalDomain(0.0,3275.0)]

S_max = domains[1].domain.upper
H_max = domains[2].domain.upper

# Boundary
bcs = [
    #p(S,H, θ) ~  (2*pi*σ_H*σ_S)^-1 * exp( -norm2(H,H0, σ_H) - norm2(S,S0, σ_S)  ),
    p(0.0,H,θ) ~ 0.,
    p(S,0.0,θ) ~ 0.,
    p(S_max,H,θ) ~ 0.,
    p(S,H_max,θ) ~ 0.,
    Dh(p(0.0,H,θ)) ~ 0.,
    Dh(p(S,0.0,θ)) ~ 0.,
    Dh(p(S_max,H,θ)) ~ 0.,
    Dh(p(S,H_max,θ)) ~ 0.,
    Ds(p(0.0,H,θ)) ~ 0.,
    Ds(p(S,0.0,θ)) ~ 0.,
    Ds(p(S_max,H,θ)) ~ 0.,
    Ds(p(S,H_max,θ)) ~ 0.,
    #p(0,H,θ) ~ p(S,0,θ),
    #p(S,0,θ) ~ p(S_max,H,θ),
    #p(S_max,H,θ) ~ p(S,H_max,θ),
    #p(S,H_max,θ) ~ p(0,H,θ)
]




# Neural Network
chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))

discretization = NeuralPDE.PhysicsInformedNN([dS,dH],
                                             chain,
                                             strategy=NeuralPDE.StochasticTraining(include_frac=0.5))

pde_system = PDESystem(eq,bcs,domains,[S, H],[p])
prob = discretize(pde_system,discretization)

@time res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb, maxiters=8000)
phi = discretization.phi
