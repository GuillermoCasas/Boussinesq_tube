using Gridap
using IterativeSolvers
using IncompleteLU
using LinearAlgebra
using AlgebraicMultigrid

import Base: \

struct NavierStokesSchurPreconditioner
    ml_A
    ilu_S
    B
    BT
    inv_diag_A
    n_u::Int
end

function NavierStokesSchurPreconditioner(M::Gridap.Algebra.BlockMatrix, n_u::Int; tau=0.01)
    A = M.blocks[1,1]
    BT = M.blocks[1,2]
    B = M.blocks[2,1]
    C = M.blocks[2,2]
    
    # State-of-the-art Algebraic Multigrid strictly bounding the Velocity Convection-Diffusion block!
    ml_A = aspreconditioner(smoothed_aggregation(A))
    
    # SIMPLE Block Schur Approximation: S = C - B * diag(A)^-1 * B^T
    inv_diag_A = 1.0 ./ diag(A)
    S = C - B * sparse(Diagonal(inv_diag_A)) * BT
    
    # Pressure Schur Poisson block efficiently bounded by IncompleteLU natively!
    ilu_S = ilu(S, τ=tau)
    
    return NavierStokesSchurPreconditioner(ml_A, ilu_S, B, BT, inv_diag_A, n_u)
end

function \(P::NavierStokesSchurPreconditioner, v::AbstractVector)
    # Forward block execution dynamically scaling the Saddle Point limits analytically
    v_u = v[1:P.n_u]
    v_p = v[P.n_u+1:end]
    
    # 1. Precondition Schur Pressure
    y_p = P.ilu_S \ v_p
    
    # 2. Update Velocity RHS
    v_u_mod = v_u - P.BT * y_p
    
    # 3. Precondition Velocity using AMG
    y_u = P.ml_A \ v_u_mod
    
    return [y_u; y_p]
end

# We import the core Gridap Algebraic framework interfaces. 
# Gridap.Algebra enforces a strict pipeline for custom linear solvers:
# 1. symbolic_setup(solver, matrix): Performs any structural matrix analysis independent of numerical values (e.g. non-zero patterns).
# 2. numerical_setup(symbolic_state, matrix): Generates the actual numerical inverse or preconditioner (e.g. ILU factorization).
# 3. solve!(x, numerical_state, b): The actual application of the Krylov method iteratively.
import Gridap.Algebra: LinearSolver, SymbolicSetup, NumericalSetup, symbolic_setup, numerical_setup, numerical_setup!, solve!

"""
    CustomIterativeSolver

A Gridap-compliant wrapper struct linking the generic abstract algebraic assembly in Gridap 
(which builds Ax = b locally element-by-element) directly to high-performance Krylov subspace 
methods contained exclusively inside `IterativeSolvers.jl`.

# Fields
- `solver_name::Symbol`: Selects the Krylov projection space (e.g., `:gmres`, `:bicgstabl`, `:cg`).
- `kwargs::Dict`: Stores solver boundary conditions like `:reltol` (relative tolerance) or `:precond` (preconditioner block strategy: :amg, :ilu, :jacobi).
"""
struct CustomIterativeSolver <: LinearSolver
    solver_name::Symbol
    kwargs::Dict
end

CustomIterativeSolver(solver_name::Symbol; kwargs...) = CustomIterativeSolver(solver_name, Dict(kwargs))

# Empty struct because we do not have specific sparsity-pattern symbolic algorithms to compute ahead of time.
struct CustomSymbolicSetup{S} <: SymbolicSetup
    solver::S
end

# Holds the active evaluated matrix `A` and its corresponding preconditioner factorization `Pl`.
mutable struct CustomNumericalSetup{S, M, P} <: NumericalSetup
    solver::S
    A::M
    Pl::P
end

function Gridap.Algebra.symbolic_setup(solver::CustomIterativeSolver, A::AbstractMatrix)
    CustomSymbolicSetup(solver)
end

"""
    build_preconditioner(solver_name::Symbol, A::AbstractMatrix, kwargs::Dict)

Constructs analytical Preconditioners P_L to improve the condition number of the discrete matrix A.
The Krylov method essentially searches P_L^{-1} A x = P_L^{-1} b, radically transforming the eigen-bound scaling.

# Preconditioner Types
1. **:ilu (Incomplete LU):** Computes an approximate direct factorization (L*U ≈ A). Works beautifully on strictly diagonal dominant systems but mathematically throws `ZeroPivotException`s structurally on saddle-point approximations (like our incompressible Boussinesq Navier-Stokes block) because the pressure degrees of freedom enforce the div(u) = 0 constraint (yielding exact 0.0 bounds natively along the central pressure diagonals).
2. **:amg (Algebraic Multigrid):** Recursively builds coarser functional representations of A, solving them symmetrically, and interpolating the smoothed residual scaling back up. Ideal for Advection-Diffusion or Poisson pressure formulations where physical interactions are purely elliptic/diffusive natively.
3. **:jacobi:** Simply takes the inverse of the diagonal components D^{-1}. Computationally essentially free, but drastically scales off scaling imbalances (like pressure scalars versus velocity vector components mapping differently out of the geometry parameters).
"""
function build_preconditioner(solver_name::Symbol, A::AbstractMatrix, kwargs::Dict)
    precond = get(kwargs, :precond, :none)
    if precond == :ilu
        # Attempt ILU. If the matrix is a saddle-point formulation (like Monolithic Boussinesq), 
        # the lack of terms directly correlating pressure points exclusively to pressure points guarantees structural 0.0 diagonals.
        τ = get(kwargs, :tau, 0.01)
        try
            return ilu(A, τ=τ)
        catch e
            println("ILU failed (often due to zero diagonals in saddle point). Falling back to Diagonal bounds.")
            d = diag(A)
            # Mathematically shield against division-by-zero during Jacobi extraction
            d[d .== 0] .= 1e-8
            return Diagonal(1.0 ./ d)
        end
    elseif precond == :amg
        try
            return aspreconditioner(smoothed_aggregation(A))
        catch e
            println("AMG failed. Falling back to Diagonal bounds.")
            d = diag(A)
            d[d .== 0] .= 1e-8
            return Diagonal(1.0 ./ d)
        end
    elseif precond == :schur
        n_u = get(kwargs, :n_u, size(A, 1) ÷ 2)
        tau = get(kwargs, :tau, 0.01) # Assuming tau might be used for Schur as well
        return NavierStokesSchurPreconditioner(A, n_u; tau=tau)
    elseif precond == :jacobi
        d = diag(A)
        d[d .== 0] .= 1e-8
        return Diagonal(1.0 ./ d)
    end
    return I # Returns Identity preconditioning structurally otherwise
end

function Gridap.Algebra.numerical_setup(ss::CustomSymbolicSetup, A::AbstractMatrix)
    Pl = build_preconditioner(ss.solver.solver_name, A, ss.solver.kwargs)
    CustomNumericalSetup(ss.solver, A, Pl)
end

function Gridap.Algebra.numerical_setup!(ns::CustomNumericalSetup, A::AbstractMatrix)
    ns.A = A
    ns.Pl = build_preconditioner(ns.solver.solver_name, A, ns.solver.kwargs)
end

"""
    solve!(x::AbstractVector, ns::CustomNumericalSetup, b::AbstractVector)

Evaluates the primary iterative Krylov root vector space algorithms finding an optimal scalar minimum.
- **GMRES:** General Minimum Residual. Stores all orthogonalized conjugate vectors dynamically minimizing arbitrarily ill-conditioned, non-symmetrical equations. (High memory cost structurally).
- **BiCGStabl:** Biconjugate Gradient Stabilized. Avoids keeping all basis vectors natively, relying strictly on bi-orthogonal inner products to solve massively asymmetric formulations (ideal for Advection components).
- **CG:** Conjugate Gradient. Only converges if matrix strictly SPD (Symmetric Positive-Definite), which our native Navier-Stokes is NOT functionally explicitly.
"""
function Gridap.Algebra.solve!(x::AbstractVector, ns::CustomNumericalSetup, b::AbstractVector)
    solver_name = ns.solver.solver_name
    kwargs_pass = copy(ns.solver.kwargs)
    delete!(kwargs_pass, :precond)
    delete!(kwargs_pass, :tau)
    delete!(kwargs_pass, :n_u)
    
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
