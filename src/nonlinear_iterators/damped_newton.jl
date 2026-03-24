using Gridap
using Gridap.Algebra: LinearSolver, NonlinearSolver, NonlinearOperator, residual, jacobian, symbolic_setup, numerical_setup, numerical_setup!
import Gridap.Algebra: solve!
using LinearAlgebra

"""
    DampedNewtonSolver <: NonlinearSolver

A custom extension of the Gridap `NonlinearSolver` interface that implements a robust Damped Newton-Raphson algorithm.
Standard Newton-Raphson attempts to find the root of the residual equation $R(x) = 0$ by linearizing the space 
via the Jacobian $J = \\frac{\\partial R}{\\partial x}$ and stepping $J \\Delta x = -R(x)$.

However, continuous physics environments (like high Reynolds/Péclet limits) produce highly nonlinear boundaries.
A standard full-step Newton method ($x_{new} = x_{old} + \\Delta x$) often diverges. 
This Damped solver implements an **Armijo Backtracking Line Search**: 
It tests fractional steps $\\alpha \\Delta x$ to guarantee that the new residual norm practically decreases before moving.

# Fields
- `ls::LinearSolver`: The underlying algebraic solver (e.g., FGMRES or BiCGStab) used to evaluate $J \\Delta x = -b$.
- `tol::Float64`: The strictly required $L^2$ outer residual norm limit.
- `max_iters::Int`: Total allowable outer Newton jumps.
- `backtrack_factor::Float64`: The scaling reduction factor $\\tau$ applied if a step fails ($0 < \\tau < 1$, typically 0.5).
- `min_alpha::Float64`: The lowest operational limit for the linesearch bound before giving up on a specific direction.
"""
struct DampedNewtonSolver <: NonlinearSolver
    ls::LinearSolver
    tol::Float64
    max_iters::Int
    backtrack_factor::Float64
    min_alpha::Float64
end

function DampedNewtonSolver(ls::LinearSolver; tol=1e-4, max_iters=20, backtrack_factor=0.5, min_alpha=1e-4)
    DampedNewtonSolver(ls, tol, max_iters, backtrack_factor, min_alpha)
end

"""
    solve!(x::AbstractVector, nls::DampedNewtonSolver, op::NonlinearOperator)

Overrides the generic `Gridap.solve!` method. 
Gridap abstracts the complex partial differential equations natively into a generic `NonlinearOperator` (`op`). 
This guarantees code isolation; the solver does not need to know the physics, it simply queries `op` for $b$ and $J$.

# Mathematical Routine
1. Evaluate $b = R(x)$ and test initial tolerance limits.
2. Evaluate local stiffness matrix $J = \\nabla_x R(x)$.
3. Solve implicitly for the direction $\\Delta x$ where $J \\Delta x = -b$.
4. Evaluate a candidate point map $x_{new} = x + \\alpha \\Delta x$. 
5. If the resulting norm $||R(x_{new})|| > ||R(x_{old})||$, recursively backtrack $\\alpha \\gets \\alpha \\cdot \\tau$.
6. Repeat until the physical space converges.
"""
function Gridap.Algebra.solve!(x::AbstractVector, nls::DampedNewtonSolver, op::NonlinearOperator)
    println("--- Starting Damped Newton-Raphson Solver ---")
    
    # Extract the discrete physical residuals evaluating the non-linear operator over DoFs
    b = residual(op, x)
    err = norm(b)
    
    if err < nls.tol
        println("Converged initially: |Residual| = $err")
        return
    end
    
    # Synthesize the fully evaluated monolithic Jacobian automatically using Gridap's AD
    J = jacobian(op, x)
    
    # Standard linear solver inheritance loop evaluating preconditioners
    ss = symbolic_setup(nls.ls, J)
    ns = numerical_setup(ss, J)
    
    dx = similar(b)
    x_new = copy(x)
    
    for iter in 1:nls.max_iters
        err = norm(b)
        println(" Newton Iteration $iter | |Residual| = ", err)
        
        if err < nls.tol
            println("Newton converged in $iter iterations.")
            return
        end
        
        # We only re-evaluate the heavy Jacobian block if we have progressed in space natively
        if iter > 1
            J = jacobian(op, x)
        end
        
        # Reset Preconditioners using structural inverse factorization
        numerical_setup!(ns, J)
        
        # Compute descent bound direction mathematically: J * dx = -R(x)
        b_neg = -b
        solve!(dx, ns, b_neg)
        
        # Initialize optimal line search bound (full 100% descent step)
        alpha = 1.0
        x_new .= x .+ alpha .* dx
        
        # Benchmark corresponding physics response
        b_new = residual(op, x_new)
        err_new = norm(b_new)
        
        # Sub-level continuous line search: Armijo sufficient decrease test natively
        while err_new > err && alpha > nls.min_alpha
            alpha *= nls.backtrack_factor # Decrease scaling (e.g. 50% split)
            x_new .= x .+ alpha .* dx
            b_new = residual(op, x_new)
            err_new = norm(b_new)
            println("   Backtracking: alpha = $alpha | |Residual| = $err_new")
        end
        
        println("  -> Step taken: alpha = $alpha | new |Residual| = $err_new")
        
        x .= x_new
        b .= b_new
    end
    println("Warning: Damped Newton solver did not converge after $(nls.max_iters) iterations.")
end
