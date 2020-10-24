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
@variables p(S,H,t)

eq = Dt(p) ~ - Dh(F_H*p) - Ds(F_S*p) + D_H * Dhh(p) + D_S *Dss(p)

# boundary conditions
bcs = [

]

domains = [ S ∈ IntervalDomain(0.0,225.0),
            H ∈ IntervalDomain(0.0,3725.0)]

# To Do 
