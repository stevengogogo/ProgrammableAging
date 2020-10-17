# Reference: https://neuralpde.sciml.ai/dev/examples/pinns_example/
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
# 3D PDE
@parameters x y t θ
@variables u1(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dt'~t

# 3D PDE
eq  = Dt(u1(x,y,t,θ)) ~ Dxx(u1(x,y,t,θ)) + Dyy(u1(x,y,t,θ))
# Initial and boundary conditions
bcs = [u1(x,y,0,θ) ~ exp(x+y)*cos(x+y) ,
       u1(0,y,t,θ) ~ exp(y)*cos(y+4t),
       u1(2,y,t,θ) ~ exp(2+y)*cos(2+y+4t) ,
       u1(x,0,t,θ) ~ exp(x)*cos(x+4t),
       u1(x,2,t,θ) ~ exp(x+2)*cos(x+2+4t)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]

# Discretization
dx = 0.25; dy= 0.25; dt = 0.25
# Neural network
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = NeuralPDE.PhysicsInformedNN([dx,dy,dt],
                                             chain,
                                             strategy = NeuralPDE.StochasticTraining(include_frac=0.9))
pde_system = PDESystem(eq,bcs,domains,[x,y,t],[u1])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1), progress = false; cb = cb, maxiters=3000)
phi = discretization.phi
