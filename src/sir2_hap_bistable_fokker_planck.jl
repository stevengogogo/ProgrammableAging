using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

using Plots
plotly()

include("OdeModel.jl")
## ODE MODEL

F_H = OdeModel.F_H
F_S = OdeModel.F_S

## Fokker planck
@parameters H, S
@parameters D_H, D_S
@parameters t
@derivatives Dt'~t
@derivatives Dss''~S
@derivatives Dhh''~H
@derivatives Ds'~S
@derivatives Dh'~H
@variables p(..)

σ_H, σ_S = 5000.0, 150.0
H0, S0 = 3275.0, 112.0

eq = Dt(p(S,H,t)) ~ - Dh(F_H*p(S,H,t)) - Ds(F_S*p(S,H,t)) + D_H * Dhh(p(S,H,t)) + D_S *Dss(p(S,H,t))

# Domains
domains = [ S ∈ IntervalDomain(0.0,225.0),
            H ∈ IntervalDomain(0.0,3725.0),
            t ∈ IntervalDomain(0.0,100.0)]

S_max = domains[1].domain.upper
H_max = domains[2].domain.upper
# boundary conditions
bcs = [
    Dt(p(0,H,t)) ~ 0.f0,
    Dt(p(S,0,t)) ~ 0.f0,
    Dt(p(S_max,H,t)) ~ 0.f0,
    Dt(p(S,H_max,t)) ~ 0.f0,
]

# Initial Value
norm2(X, a, b) = (X-a)^2 / b^2

p0(h,s) = (2*pi*σ_H*σ_S)^-1 * exp( -norm2(h,H0, σ_H) - norm2(s,S0, σ_S)  )

# tspan
tspan = (0.0f0, 256.0f0)



# Discretization
dx = 1

# Nerual Network
dim = 3

chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1)) |> gpu


discretization = PhysicsInformedNN(dx, chain)

pde_system = PDESystem(eq, bcs, domains, [S, H, t], [p])

@time prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb, maxiters=1000)
phi = discretization.phi




## Create Terminal PDE Problem



TerminalPDEProblem
