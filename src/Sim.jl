module Sim

    using DifferentialEquations 


    function solveT(eqs, tspan)
        sys = ODESystem(eqs)
        tspan = (0.0,100.0)

        prob = ODEProblem(sys,u0,tspan,p,jac=true)
        sol = solve(prob)

        return sol
    end

    function solveNullcline(eqs)


        ode_f = ODEFunction(sys)

        substitute.(eqs[1].rhs,p)
    end

end
