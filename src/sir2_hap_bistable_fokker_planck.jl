using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

using Plots
plotly()

include("OdeModel.jl")
## ODE MODEL
hill = OdeModel.hill
@parameters t k₁ k₂ k₃ k₄ k₅ k₆ KM₁ KM₂ KM₃ KM₄ n₁ n₂ n₃ n₄ α β S_tot

## Fokker planck
@parameters H, S, θ
@parameters D_H, D_S
@derivatives Dt'~t
@derivatives Dss''~S
@derivatives Dhh''~H
@derivatives Ds'~S
@derivatives Dh'~H
@variables p(..)

F_H = k₁ * (1-α)*hill(S, KM₁, n₁) * hill(H, KM₂, n₂) + k₂ - k₃*H
F_S = k₄ * (1-β)*hill(H, KM₃,n₃) * hill(S, KM₄, n₄)*(S_tot - S) + k₅ - k₆*S



# Replace with number
p_ =[k₁ => 40,
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
    S_tot => 225,
    D_H=>500,
    D_S=>0.45]

F_H = substitute.(F_H, (p_,))[1]
F_S = substitute.(F_S, (p_,))[1]




σ_H, σ_S = 5000.0, 150.0
H0, S0 = 3275.0, 112.0

eq = Dt(p(S,H,t,θ)) ~ - Dh(F_H*p(S,H,t,θ)) - Ds(F_S*p(S,H,t,θ)) + 500.0 * Dhh(p(S,H,t,θ)) + 0.45 *Dss(p(S,H,t,θ))

# Domains
domains = [ S ∈ IntervalDomain(0.0,112.0),
            H ∈ IntervalDomain(0.0,3275.0),
            t ∈ IntervalDomain(0.0,300.0)]
S_max = domains[1].domain.upper
H_max = domains[2].domain.upper

# boundary conditions
norm2(X, a, b) = (X-a)^2 / b^2
bcs = [
    p(S,H, 0, θ) ~  (2*pi*σ_H*σ_S)^-1 * exp( -norm2(H,H0, σ_H) - norm2(S,S0, σ_S)  ),
    p(0,H,t,θ) ~ 0.f0,
    p(S,0,t,θ) ~ 0.f0,
    p(S_max,H,t,θ) ~ 0.f0,
    p(S,H_max,t,θ) ~ 0.f0,
]


# Discretization
dS = 10.0; dH=100.0; dt = 10.0





# Nerual Network
dim = 1

chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

initθ = initial_params(chain)

discretization = NeuralPDE.PhysicsInformedNN([dS,dH,dt],
                                             chain,
                                             initθ,
                                             strategy = NeuralPDE.StochasticTraining(include_frac=0.9))

pde_system = PDESystem(eq,bcs,domains,[S,H,t],[p])


prob = NeuralPDE.discretize(pde_system,discretization)


cb = function (p,l)
    println("Current loss is: $l")
    return false
end

@time res = GalacticOptim.solve(prob, ADAM(0.1), progress = false; cb = cb, maxiters=3000)

phi = discretization.phi




## Create Terminal PDE Problem



TerminalPDEProblem
