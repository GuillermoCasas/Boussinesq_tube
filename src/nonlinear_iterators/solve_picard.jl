using Gridap
using LinearAlgebra
using GridapPETSc
using GridapSolvers
using GridapSolvers.LinearSolvers

"""
    solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g)

Executes a conditionally stable, staggered operator-splitting Picard solver for the Boussinesq equations.
Rather than solving 3x3 (u, p, c) simultaneously (which creates highly ill-conditioned saddle points), 
we decouple the physics sequentially:

1. **Given previous states** u_prev, c_prev.
2. **Solve Navier-Stokes** (u, p): Linearize the convective maps ((u_prev . \\nabla) u) and fix the buoyancy force based on c_prev.
3. **Solve Advection-Diffusion** (c): Feed the newly solved velocity u_new strictly as an explicit flow field (u_new . \\nabla) c.
4. **Relativize Changes**: If the step difference ||u_new - u_prev|| / ||u_new|| < tol, exit structurally.

# Streamline Upwind Petrov-Galerkin (SUPG) Stabilization (\\tau_m, \\tau_c)
Because pure Galerkin finite elements fail and produce massive node-to-node oscillations at high Péclet/Reynolds numbers (convection-dominant flows), 
we bias the test formulation physically upwind. The \\tau terms evaluate the mesh size h and velocity scalar norm to construct exact artificial element diffusion arrays structurally parallel to the analytical velocity field.
"""
function _picard_inner_loop(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, solver_NS, solver_AD)
    rho0 = phys_params.rho0
    mu = phys_params.mu
    D = phys_params.D
    g_mag = phys_params.g_mag
    beta_c = phys_params.beta_c
    c_ref = phys_params.c_ref
    e_g = phys_params.e_g
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
        
        # Establish the Algebraic Artificial Compressibility constant `ϵ` dynamically.
        # Natively appending `ϵ * p * q` across both the LHS matrix array and evaluating it linearly over `ϵ * p_prev * q` on the RHS safely preserves exact spatial continuity limit on Convergence!
        ϵ = 1e-8
        
        # Formulate linearized Affine Weak formulation (Bilinear 'a_NS', linear 'l_NS')
        a_NS(X, Y) = ∫( 
            rho0 * (uh_prev ⋅ ∇(X[1])) ⋅ Y[1] + # Convection
            2 * mu * (ε(X[1]) ⊙ ε(Y[1])) -      # Diffusion (Stress strain tensor)
            X[2] * (∇ ⋅ Y[1]) +                 # Pressure divergence gradient
            Y[2] * (∇ ⋅ X[1]) +                 # Incompressibility constraint
            ϵ * X[2] * Y[2] +                   # ϵ Artificial Compressibility matrix map boundary explicitly required natively for ILU
            # SUPG formulation testing along the streamline
            τ_m_field * (rho0 * (uh_prev ⋅ ∇(Y[1]))) ⋅ (rho0 * (uh_prev ⋅ ∇(X[1])) + ∇(X[2]))
        )dΩ
        
        l_NS(Y) = ∫( 
            body_force ⋅ Y[1] + 
            τ_m_field * (rho0 * (uh_prev ⋅ ∇(Y[1]))) ⋅ body_force +
            ϵ * ph_prev * Y[2]                  # Exact mathematical formulation recovery restoring precise 𝒪(h^3) dynamics
        )dΩ
        
        local op_NS
        @timeit "Assemble Navier-Stokes Matrix" begin
            op_NS = AffineFEOperator(a_NS, l_NS, X_NS, Y_NS)
        end
        @timeit "Solve Navier-Stokes System" begin
            uh_new, ph_new = Gridap.solve(solver_NS, op_NS)
        end
        
        # Advection-Diffusion Block: Update SUPG variables instantly with the freshly solved velocity bounds
        τ_c_field = τ_c ∘ uh_new
        
        a_AD(c, w) = ∫( 
            (uh_new ⋅ ∇(c)) * w +               # Convection mapping
            D * (∇(c) ⋅ ∇(w)) +                 # Solute thermal diffusion
            τ_c_field * (uh_new ⋅ ∇(w)) * (uh_new ⋅ ∇(c)) # SUPG stabilization along the streamline
        )dΩ
        
        l_AD(w) = ∫( 0.0 * w )dΩ
        
        local op_AD
        @timeit "Assemble Advection-Diffusion Matrix" begin
            op_AD = AffineFEOperator(a_AD, l_AD, Uc, Vc)
        end
        @timeit "Solve Advection-Diffusion System" begin
            ch_new = Gridap.solve(solver_AD, op_AD)
        end
        
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

function solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params)
    ns_cfg = config["numerical"]["solver_NS"]
    
    n_u = Gridap.FESpaces.num_free_dofs(Uu)

    # Fully abstracted pure Julia environment utilizing native block algebraic mappings
    if ns_cfg["type"] == "petsc" || ns_cfg["type"] == "petsc_amg"
        solver_NS = PETScLinearSolver()
    elseif ns_cfg["type"] == "lu"
        solver_NS = LUSolver()
    elseif ns_cfg["type"] == "GridapSolvers"
        # 1. State-of-the-art velocity scalar preconditioning natively evaluated by Algebraic Multigrid limits
        solver_u = CustomIterativeSolver(:bicgstabl; precond=:amg, reltol=1e-4)
        
        # 2. Approximate Mass Matrix Schur limits identically locally
        solver_p = CustomIterativeSolver(:cg; precond=:jacobi, reltol=1e-4)

        bblocks  = [GridapSolvers.BlockSolvers.LinearSystemBlock()    GridapSolvers.BlockSolvers.LinearSystemBlock();
                    GridapSolvers.BlockSolvers.LinearSystemBlock()    GridapSolvers.BlockSolvers.BiformBlock((p,q) -> ∫( 1.0 * p * q )dΩ, Up, Up)]
        
        coeffs = [1.0 1.0;
                  0.0 1.0]  
                  
        P = GridapSolvers.BlockSolvers.BlockTriangularSolver(bblocks, [solver_u, solver_p], coeffs, :upper)
        solver_NS = GridapSolvers.LinearSolvers.FGMRESSolver(30, P; atol=1e-10, rtol=1e-4, verbose=true)
    else
        solver_NS = CustomIterativeSolver(Symbol(ns_cfg["type"]); precond=Symbol(ns_cfg["precond"]), reltol=get(ns_cfg, "reltol", 1e-4), tau=get(ns_cfg, "tau", 0.01), n_u=n_u)
    end
    
    # The Advection-Diffusion block solver is independent of the Navier-Stokes one
    ad_cfg = config["numerical"]["solver_AD"]
    if get(ad_cfg, "type", "") == "petsc" || get(ad_cfg, "type", "") == "petsc_amg"
        solver_AD = PETScLinearSolver()
    elseif ad_cfg["type"] == "lu"
        solver_AD = LUSolver()
    else
        solver_AD = CustomIterativeSolver(Symbol(ad_cfg["type"]); precond=Symbol(ad_cfg["precond"]), reltol=get(ad_cfg, "reltol", 1e-4), tau=get(ad_cfg, "tau", 0.01))
    end
    
    return _picard_inner_loop(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, solver_NS, solver_AD)
end
