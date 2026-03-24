using Gridap
using LinearAlgebra

function solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g)
    tol = config["numerical"]["tol"]
    max_iters = config["numerical"]["max_iters"]

    uh_prev = interpolate_everywhere(zero(e_g), Uu)
    ph_prev = interpolate_everywhere(0.0, Up)
    ch_prev = interpolate_everywhere(c_in_func, Uc)

    uh_new = uh_prev
    ph_new = ph_prev
    ch_new = ch_prev

    for iter in 1:max_iters
        println("--- Iteration $iter ---")
        
        τ_m_field = τ_m ∘ uh_prev
        body_force = rho0 * g_mag * beta_c * (ch_prev - c_ref) * e_g
        
        a_NS(X, Y) = ∫( 
            rho0 * (uh_prev ⋅ ∇(X[1])) ⋅ Y[1] + 
            2 * mu * (ε(X[1]) ⊙ ε(Y[1])) - 
            X[2] * (∇ ⋅ Y[1]) + 
            Y[2] * (∇ ⋅ X[1]) + 
            τ_m_field * (rho0 * (uh_prev ⋅ ∇(Y[1]))) ⋅ (rho0 * (uh_prev ⋅ ∇(X[1])) + ∇(X[2]))
        )dΩ
        
        l_NS(Y) = ∫( 
            body_force ⋅ Y[1] + 
            τ_m_field * (rho0 * (uh_prev ⋅ ∇(Y[1]))) ⋅ body_force 
        )dΩ
        
        op_NS = AffineFEOperator(a_NS, l_NS, X_NS, Y_NS)
        ns_cfg = config["numerical"]["solver_NS"]
        solver_NS = CustomIterativeSolver(Symbol(ns_cfg["type"]); precond=Symbol(ns_cfg["precond"]), reltol=ns_cfg["reltol"])
        uh_new, ph_new = Gridap.solve(solver_NS, op_NS)
        
        τ_c_field = τ_c ∘ uh_new
        
        a_AD(c, w) = ∫( 
            (uh_new ⋅ ∇(c)) * w + 
            D * (∇(c) ⋅ ∇(w)) + 
            τ_c_field * (uh_new ⋅ ∇(w)) * (uh_new ⋅ ∇(c))
        )dΩ
        
        l_AD(w) = ∫( 0.0 * w )dΩ
        
        op_AD = AffineFEOperator(a_AD, l_AD, Uc, Vc)
        ad_cfg = config["numerical"]["solver_AD"]
        solver_AD = CustomIterativeSolver(Symbol(ad_cfg["type"]); precond=Symbol(ad_cfg["precond"]), tau=ad_cfg["tau"], reltol=ad_cfg["reltol"])
        ch_new = Gridap.solve(solver_AD, op_AD)
        
        du_norm = norm(get_free_dof_values(uh_new) .- get_free_dof_values(uh_prev))
        dc_norm = norm(get_free_dof_values(ch_new) .- get_free_dof_values(ch_prev))
        
        u_norm_tot = norm(get_free_dof_values(uh_new)) + 1e-10
        c_norm_tot = norm(get_free_dof_values(ch_new)) + 1e-10
        
        err_u = du_norm / u_norm_tot
        err_c = dc_norm / c_norm_tot
        
        println(" Relative error U: $(round(err_u, sigdigits=4))")
        println(" Relative error C: $(round(err_c, sigdigits=4))")
        
        if err_u < tol && err_c < tol
            println("Converged in $iter iterations!")
            break
        end
        
        uh_prev = uh_new
        ph_prev = ph_new
        ch_prev = ch_new
    end
    
    return uh_new, ph_new, ch_new
end
