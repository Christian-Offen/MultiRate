# Julia implementation of the multirate integration scheme used in the numerical experiments of the chapter
# Leyendecker S., Ober-Blöbaum S. (2013)
# A Variational Approach to Multirate Integration for Constrained Systems.
# In: Samin JC., Fisette P. (eds) Multibody Dynamics. Computational Methods in Applied Sciences, vol 28. Springer, Dordrecht. 
# https://doi.org/10.1007/978-94-007-5404-1_5


using Flux: gradient
import Base.Iterators: flatten
using NLsolve
#using ProgressBars


function BuildDiscreteL(fastPotential, slowPotential, kineticTerm)
    # returns discrete multi-rate Lagrangian given continuous functions for the fast potential, the slow potential, the kinetic term
    
    function qsInterpolate(qs0,qs1,m,h,microsteps)
    1/microsteps*((microsteps-m)*qs0 + m*qs1)
    end

    function DiscreteSlowPotential(qs0,qs1,qf0,h,microsteps)                             # qf0 is matrix with dimension (microsteps,length(qf))
    	h_micro = h/microsteps  # length micro step
        h_micro*sum(map(m->slowPotential(qsInterpolate(qs0,qs1,m-1,h,microsteps),qf0[m,:]),1:microsteps))
    end

    function DiscreteFastPotential(qf0,h,microsteps)
        qfmp=(qf0[1:end-1,:]+qf0[2:end,:])/2

        # rewriting mapslices for Zygote
        s = 0.
        for j=1:microsteps
            s += fastPotential(qfmp[j,:])
        end

	h_micro = h/microsteps  # length micro step

        return h_micro*s

    end


    function DiscreteKineticTerm(qs0,qs1,qf0,h,microsteps)
        
        h_micro = h/microsteps  # length micro step
        
        qsdot = (qs1-qs0)/h
        qfdot = (qf0[2:end,:]-qf0[1:end-1,:])/h_micro

        #rewriting mapslices for Zygote
        s = 0.
        for j=1:microsteps
            s += kineticTerm(qsdot,qfdot[j,:])
        end

        return h_micro*s

    end

    function Ld(qs0,qs1,qf0,h,microsteps)
        DiscreteKineticTerm(qs0,qs1,qf0,h,microsteps) - DiscreteSlowPotential(qs0,qs1,qf0,h,microsteps) - DiscreteFastPotential(qf0,h,microsteps)
    end
    
    return Ld
end




function MultiRateIntegrator(Ld)
    # returns the function MultiRateIntegrate as well as InitializationValues, and MultiRateStep for a discrete multirate Lagrangian Ld.
    
    


    function InitializationValues(qs0,qf00,ps0,pf00,h,microsteps) # (qs0,qf00),(ps0,pf00) initial configuration
	# compute the initial step from initial configuration

        n = length(qf00)

        function InitHelper(qs1,qf0Tail)    # qf0Tail is matrix with dimension (microsteps,length(qf)) (leave out microstep at t_k^0 but include t_k^microstep)
	    # need to solve InitHelper(qs1,qf0tail)==0 to determine initialisation values from initial configuration
            qf0Inner = qf0Tail[1:end-1,:]
            qf0End   = qf0Tail[end,:]

            partial_Ld(qs0,qf00,qf0Inner) = Ld(qs0,qs1,vcat(transpose(qf00),qf0Inner,transpose(qf0End)),h,microsteps)

            dLd0 = gradient( partial_Ld, qs0,qf00,qf0Inner)
            dLd0[1] .+= ps0
            dLd0[2] .+= pf00

            return collect(flatten(dLd0))

        end

        function InitHelper(A)
            InitHelper(A[1:n],reshape(A[n+1:end],(microsteps,n)))
        end

        initsSol = nlsolve(InitHelper, zeros((microsteps+1)*n),autodiff = :forward)

        inits = initsSol.zero

        # distribute output to qs1 and inner microstep points qf0
        qs1 = inits[1:n]
        qf0Tail = reshape(inits[n+1:end],(microsteps,n))

        qf0=vcat(transpose(qf00),qf0Tail);


        return qs1, qf0,initsSol

    end            
    
    
    function MultiRateStep(qs0,qs1,qf0,h,microsteps)
	# computes a step with the multi-rate integrator
	
        n = length(qs0)

        qf10 = qf0[end,:]

        function objective(qs2,qf1Tail) 

            qf1Inner = qf1Tail[1:end-1,:]
            qf1End = qf1Tail[end,:]

            # attention: qf10 occurs in qf0 as last element and in qf1 as first element
            Ld_partial(qs1,qf10,qf1Inner) = Ld(qs1,qs2,vcat(transpose(qf10),qf1Inner,transpose(qf1End)),h,microsteps) + Ld(qs0,qs1,vcat(qf0[1:end-1,:],transpose(qf10)),h,microsteps)

            return gradient(Ld_partial, qs1,qf10,qf1Inner)
        end

        objective(A) = collect(flatten(objective(A[1:n],reshape(A[n+1:end],(microsteps,n)))))

        Aguess = vcat(qs1, reshape(qf0[2:end,:],microsteps*n))

        qnextSol = nlsolve(objective,Aguess,autodiff = :forward)

        # distribute to variables
        qs2     = qnextSol.zero[1:n]
        qf1Tail = reshape(qnextSol.zero[n+1:end],(microsteps,n))
        qf1 = vcat(transpose(qf0[end,:]),qf1Tail)


        return qs2,qf1,qnextSol

    end
    
    
    function MultiRateIntegrate(qs0,qf00,ps0,pf00,N,h,microsteps)
    
    	# returns slow and fast variables for N steps with the multi-rate integrator
    	# qs0,qf00,ps0,pf00 initial configuration
    	# N number of steps
    	# h lenggth macro step
    	# microsteps number of microsteps
        
        # compute 1st macro step
        qs1,qf0,initsSol=InitializationValues(qs0,qf00,ps0,pf00,h,microsteps)
        
        # pre-allocate memory
        QS = Array{Float64}(undef,(length(qs0),N+1))
        QF = Array{Float64}(undef,(length(qs0),N*microsteps+1))
        checkSolver = Array{Bool, 1}(undef, N-1)

        QS[:,1] = qs0
        QS[:,2] = qs1
        QF[:,1:microsteps+1] = transpose(qf0);
        
        for j = 2:N #ProgressBar(2:N) 
    
            # computation of q_j

            qsk0 = QS[:,j-1]
            qsk1 = QS[:,j]
            qfk0 = QF[:,(j-2)*microsteps+1:(j-1)*microsteps+1]

            qsk2,qfk1, chkSolver = MultiRateStep(qsk0,qsk1,transpose(qfk0),h,microsteps)

            QS[:,j+1] = qsk2
            QF[:,(j-1)*microsteps+1:j*microsteps+1] = transpose(qfk1)
            checkSolver[j-1] = chkSolver.f_converged

        end

        return QS,QF,checkSolver
        
    end
    
    

    return MultiRateIntegrate, InitializationValues, MultiRateStep
    
end



function DiscreteConjugateMomenta(Ld,QS,QF,h,microsteps)
    
    # compute discrete conjugate momenta on macro-grid
    # Ld discrete Lagrangian
    # QS slow variables, Matrix of dimension (dimension of qs variable,number of macro-steps)
    # QF fast variables, Matrix of dimension (dimension of qf variable,number of micro-steps)
    
    dLd_qs0(qs0,qs1,qf0)  = gradient(qs0 -> Ld(qs0,qs1,qf0,h,microsteps),qs0)[1]
    dLd_qs1(qs0,qs1,qf0)  = gradient(qs1 -> Ld(qs0,qs1,qf0,h,microsteps),qs1)[1]
    dLd_qf00(qs0,qs1,qf0) = gradient(qf00 -> Ld(qs0,qs1,vcat(transpose(qf00),qf0[2:end,:]),h,microsteps),qf0[1,:])[1]
    dLd_qf0end(qs0,qs1,qf0) = gradient(qf0end -> Ld(qs0,qs1,vcat(qf0[1:end-1,:],transpose(qf0end)),h,microsteps),qf0[end,:])[1]
    
    microsteps = convert(Int64,(size(QF)[2]-1)/(size(QS)[2]-1))
    
    # pre-allocate memory
    PS = Array{Float64}(undef,size(QS))
    PF = Array{Float64}(undef,size(QS))
    
    # compute discrete conjugate momenta
    for j=1:(size(QS)[2]-1)
        
        qfj = transpose(QF[:,(j-1)*microsteps+1:j*microsteps+1])
        
    
        PS[:,j]   = -dLd_qs0(QS[:,j],QS[:,j+1],qfj)
        PS[:,j+1] = dLd_qs1(QS[:,j],QS[:,j+1],qfj)                    # only very last component will not get overwritten, excellent jit-compiler will notice
        PF[:,j]   = -transpose(dLd_qf00(QS[:,j],QS[:,j+1],qfj))
        PF[:,j+1] = transpose(dLd_qf0end(QS[:,j],QS[:,j+1],qfj))      # only very last component will not get overwritten, excellent jit-compiler will notice
        
    end
    
    
    
    
    return PS,PF
    
    
end

