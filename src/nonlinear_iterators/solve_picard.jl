using Gridap
using LinearAlgebra

"""
    solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g)

Executes a conditionally stable, staggered operator-splitting Picard solver for the Boussinesq equations.
Rather than solving $3 \\times 3$ $(u, p, c)$ simultaneously (which creates highly ill-conditioned saddle points), 
we decouple the physics sequentially:

1. **Given previous states** $u_{prev}, c_{prev}$.
2. **Solve Navier-Stokes** $(u, p)$: Linearize the convective maps $((u_{prev} \\cdot \\nabla) u)$ and fix the buoyancy force based on $c_{prev}$.
3. **Solve Advection-Diffusion** $(c)$: Feed the newly solved velocity $u_{new}$ strictly as an explicit flow field $(u_{new} \\cdot \\nabla) c$.
4. **Relativize Changes**: If the step difference $||u_{new} - u_{prev}|| / ||u_{new}|| < tol$, exit structurally.

# Streamline Upwind Petrov-Galerkin (SUPG) Stabilization ($\\tau_m, \\tau_c$)
Because pure Galerkin finite elements fail and produce massive node-to-node oscillations at high Péclet/Reynolds numbers (convection-dominant flows), 
we bias the test formulation physically upwind. The $\\tau$ terms evaluate the mesh size $h$ and velocity scalar norm to construct exact artificial element diffusion arrays structurally parallel to the analytical velocity field.
"""
function solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g)
    tol = config["numerical"]["tol"]
    max_iters = config["numerical"]["max_iters"]

    # Initialize zero-bound previous iterative fields
    uh_prev = interpolate_everywhere(zero(e_g), Uu)
    ph_prev = interpolate_everywhere(0.0, Up)
    ch_prev = interpolate_everywhere(c_in_func, Uc)

    uh_new = uh_prev
    ph_new = ph_prev
    ch_new = ch_prev

    for iter in 1:max_iters
        println("--- Iteration $iter ---")
        
        # Navier-Stokes Block: Extract SUPG maps strictly from the prior velocity evaluation
        τ_m_field = τ_m ∘ uh_prev
        
        # Lagged Buoyancy: Assumes thermal/solute field is effectively constant from the prior step
        body_force = rho0 * g_mag * beta_c * (ch_prev - c_ref) * e_g
        
        # Formulate linearized Affine Weak formulation (Bilinear 'a_NS', linear 'l_NS')
        a_NS(X, Y) = ∫( 
            rho0 * (uh_prev ⋅ ∇(X[1])) ⋅ Y[1] + # Convection
            2 * mu * (ε(X[1]) ⊙ ε(Y[1])) -      # Diffusion (Stress strain tensor)
            X[2] * (∇ ⋅ Y[1]) +                 # Pressure divergence gradient
            Y[2] * (∇ ⋅ X[1]) +                 # Incompressibility constraint
            # SUPG formulation testing along the streamline
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
        
        # Advection-Diffusion Block: Update SUPG variables instantly with the freshly solved velocity bounds
        τ_c_field = τ_c ∘ uh_new
        
        a_AD(c, w) = ∫( 
            (uh_new ⋅ ∇(c)) * w +               # Convection mapping
            D * (∇(c) ⋅ ∇(w)) +                 # Solute thermal diffusion
            τ_c_field * (uh_new ⋅ ∇(w)) * (uh_new ⋅ ∇(c)) # SUPG stabilization along the streamline
        )dΩ
        
        l_AD(w) = ∫( 0.0 * w )dΩ
        
        op_AD = AffineFEOperator(a_AD, l_AD, Uc, Vc)
        ad_cfg = config["numerical"]["solver_AD"]
        solver_AD = CustomIterativeSolver(Symbol(ad_cfg["type"]); precond=Symbol(ad_cfg["precond"]), tau=ad_cfg["tau"], reltol=ad_cfg["reltol"])
        ch_new = Gridap.solve(solver_AD, op_AD)
        
        # Calculate strict relative displacement arrays across all geometric Nodes to assess step bounds 
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
        
        # Cycle back pointers
        uh_prev = uh_new
        ph_prev = ph_new
        ch_prev = ch_new
    end
    
    return uh_new, ph_new, ch_new
end
