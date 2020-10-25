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


# Discretization
dx = 1

σ_H, σ_S = 5000.0, 150.0
H0, S0 = 3275.0, 112.0

eq = [0. ~ - Dh(F_H*p(S,H,θ)) - Ds(F_S*p(S,H,θ)) + 500.0 * Dhh(p(S,H,θ)) + 0.45 *Dss(p(S,H,θ)),
      dx * p(S,H,θ) ~ 1.]

# Domains
domains = [ S ∈ IntervalDomain(0.0,225.0),
            H ∈ IntervalDomain(0.0,3725.0)]
S_max = domains[1].domain.upper
H_max = domains[2].domain.upper
# boundary conditions
bcs = [
    p(0,H) ~ 0.f0,
    p(S,0) ~ 0.f0,
    p(S_max,H) ~ 0.f0,
    p(S,H_max) ~ 0.f0,
]

# Initial Value
norm2(X, a, b) = (X-a)^2 / b^2

p0(h,s) = (2*pi*σ_H*σ_S)^-1 * exp( -norm2(h,H0, σ_H) - norm2(s,S0, σ_S)  )






# Nerual Network
dim = 1

chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))


discretization = PhysicsInformedNN(dx, chain,
                        strategy= NeuralPDE.GridTraining())

pde_system = PDESystem(eq, bcs, domains, [S, H], [p])

@time prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb, maxiters=1000)
phi = discretization.phi




## Create Terminal PDE Problem



TerminalPDEProblem
