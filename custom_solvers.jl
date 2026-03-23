using Gridap
using IterativeSolvers
using IncompleteLU
using LinearAlgebra
using AlgebraicMultigrid

import Gridap.Algebra: LinearSolver, SymbolicSetup, NumericalSetup, symbolic_setup, numerical_setup, numerical_setup!, solve!

struct CustomIterativeSolver <: LinearSolver
    solver_name::Symbol
    kwargs::Dict
end

CustomIterativeSolver(solver_name::Symbol; kwargs...) = CustomIterativeSolver(solver_name, Dict(kwargs))

struct CustomSymbolicSetup{S} <: SymbolicSetup
    solver::S
end

mutable struct CustomNumericalSetup{S, M, P} <: NumericalSetup
    solver::S
    A::M
    Pl::P
end

function Gridap.Algebra.symbolic_setup(solver::CustomIterativeSolver, A::AbstractMatrix)
    CustomSymbolicSetup(solver)
end

function build_preconditioner(solver_name::Symbol, A::AbstractMatrix, kwargs::Dict)
    precond = get(kwargs, :precond, :none)
    if precond == :ilu
        # ILU might fail for zero-diagonal (saddle point like NS)
        # Adding small shift to diagonal for ILU factorization if needed, or just standard ILU
        τ = get(kwargs, :tau, 0.01)
        try
            return ilu(A, τ=τ)
        catch e
            println("ILU failed (often due to zero diagonals in saddle point). Falling back to Diagonal.")
            d = diag(A)
            # Replace structural zeros with small number
            d[d .== 0] .= 1e-8
            return Diagonal(1.0 ./ d)
        end
    elseif precond == :amg
        try
            return aspreconditioner(smoothed_aggregation(A))
        catch e
            println("AMG failed. Falling back to Diagonal.")
            d = diag(A)
            d[d .== 0] .= 1e-8
            return Diagonal(1.0 ./ d)
        end
    elseif precond == :jacobi
        d = diag(A)
        d[d .== 0] .= 1e-8
        return Diagonal(1.0 ./ d)
    end
    return I
end

function Gridap.Algebra.numerical_setup(ss::CustomSymbolicSetup, A::AbstractMatrix)
    Pl = build_preconditioner(ss.solver.solver_name, A, ss.solver.kwargs)
    CustomNumericalSetup(ss.solver, A, Pl)
end

function Gridap.Algebra.numerical_setup!(ns::CustomNumericalSetup, A::AbstractMatrix)
    ns.A = A
    ns.Pl = build_preconditioner(ns.solver.solver_name, A, ns.solver.kwargs)
end

function Gridap.Algebra.solve!(x::AbstractVector, ns::CustomNumericalSetup, b::AbstractVector)
    solver_name = ns.solver.solver_name
    kwargs_pass = copy(ns.solver.kwargs)
    delete!(kwargs_pass, :precond)
    delete!(kwargs_pass, :tau)
    
    kwargs_pass[:log] = true
    kwargs_pass[:verbose] = true
    
    if solver_name == :gmres
        x_new, hist = gmres(ns.A, b; Pl=ns.Pl, kwargs_pass...)
        println("\n>>> Solver $solver_name finished | Iterations: ", hist.iters, " | Converged: ", hist.isconverged)
        x .= x_new
    elseif solver_name == :cg
        x_new, hist = cg(ns.A, b; Pl=ns.Pl, kwargs_pass...)
        println("\n>>> Solver $solver_name finished | Iterations: ", hist.iters, " | Converged: ", hist.isconverged)
        x .= x_new
    elseif solver_name == :bicgstabl
        x_new, hist = bicgstabl(ns.A, b; Pl=ns.Pl, kwargs_pass...)
        println("\n>>> Solver $solver_name finished | Iterations: ", hist.iters, " | Converged: ", hist.isconverged)
        x .= x_new
    else
        error("Unknown solver $solver_name")
    end
end
