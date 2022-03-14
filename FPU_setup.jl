function FPU(omega)

# returns functions for the fast potential, the slow potential, kineticTerm, Lagragian, Hamiltonian and Oscillatory Energy
# of the FPUT problem, when provided with the stiffness parameter omega

# FPUT problem: see I.5.1 A Fermi–Pasta–Ulam Problem in Geometric Numerical Integration by Hairer, Lubich, Wanner (2006)
    
    function fastPotential(qf)
    omega^2/2*sum(qf.^2)
    end

    function slowPotential(qs,qf)
        qd= qs-qf
        1/4*(qd[1]^4 + sum((qd[2:end]-qs[1:end-1]-qf[1:end-1]).^4) + qd[end]^4)
    end

    function kineticTerm(qsdot,qfdot)
        1/2*(sum(qsdot.^2)+sum(qfdot.^2))
    end

    function L(qs,qf,qsdot,qfdot)
        kineticTerm(qsdot,qfdot)-slowPotential(qs,qf)-fastPotential(qf)
    end

    function Hamiltonian(qs,qf,qsdot,qfdot)
        kineticTerm(qsdot,qfdot)+slowPotential(qs,qf)+fastPotential(qf)
    end
    
    function OscillatoryEnergy(qf,pf)
    # adiabatic invariant
        return sum(1/2*(pf.^2 .+omega^2*qf.^2),dims=1)
    end
    
    
    return fastPotential, slowPotential, kineticTerm, L, Hamiltonian, OscillatoryEnergy
    
end

