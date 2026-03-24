using Gridap
using Gridap.Algebra: LinearSolver, NonlinearSolver, NonlinearOperator, residual, jacobian, symbolic_setup, numerical_setup, numerical_setup!
import Gridap.Algebra: solve!
using LinearAlgebra

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

function Gridap.Algebra.solve!(x::AbstractVector, nls::DampedNewtonSolver, op::NonlinearOperator)
    println("--- Starting Damped Newton-Raphson Solver ---")
    b = residual(op, x)
    err = norm(b)
    
    if err < nls.tol
        println("Converged initially: |Residual| = $err")
        return
    end
    
    J = jacobian(op, x)
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
        
        if iter > 1
            J = jacobian(op, x)
        end
        
        numerical_setup!(ns, J)
        
        b_neg = -b
        solve!(dx, ns, b_neg)
        
        alpha = 1.0
        x_new .= x .+ alpha .* dx
        
        b_new = residual(op, x_new)
        err_new = norm(b_new)
        
        while err_new > err && alpha > nls.min_alpha
            alpha *= nls.backtrack_factor
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
