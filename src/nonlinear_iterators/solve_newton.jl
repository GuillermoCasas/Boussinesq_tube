using Gridap
using GridapSolvers
using GridapPETSc

"""
    solve_boussinesq_newton(config, X_coup, Y_coup, dΩ, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g, Up=nothing, Vp=nothing)

Executes a fully coupled, monolithic Damped Newton-Raphson approximation of the Boussinesq equations.
Instead of decoupling the physics iteratively, this function formulates all variables x = (u, p, c) into a 
single, massive 3x3 root-finding residual operator.

# The `res_coupled` Mathematical Operator
The operator sums the stabilization techniques and fundamental equations:
1. **Navier-Stokes (`r_ns`)**: Includes the convective inertia (u . \\nabla)u, viscous momentum \\mu \\Delta u, pressure gradient \\nabla p, continuity constraint \\nabla . u = 0, and buoyancy forcing \\rho_0 g \\beta_c (c - c_ref). It includes an SUPG (Streamline Upwind Petrov-Galerkin) component scaled by \\tau_m to suppress high-Reynolds oscillations natively.
2. **Advection-Diffusion (`r_ad`)**: Contains the convective map u . \\nabla c and thermal/solutal diffusion D \\Delta c, again stabilized via \\tau_c SUPG maps.

# The Preconditioning Challenge (Schur Complements)
The resulting 3x3 Saddle Point Jacobian is severely ill-conditioned. The continuity equation 
\\int q (\\nabla . u) structurally places exact 0.0 bounds along the main pressure diagonal. Thus, symmetric solvers fail to find an inverse direction.

We mitigate this by decomposing the discrete equations into a **Right Block-Triangular Preconditioner** P_R:
[ A  B^T ]
[ 0 -S_tilde ]
where A wraps the heavily coupled (u, c) convective fluxes, B links the pressure gradients, and S_tilde 
is our approximation for the Schur Complement (ideally S = B A^{-1} B^T).
"""
function solve_boussinesq_newton(config, X_coup, Y_coup, dΩ, τ_m, τ_c, phys_params, Up=nothing, Vp=nothing)
    rho0 = phys_params.rho0
    mu = phys_params.mu
    D = phys_params.D
    g_mag = phys_params.g_mag
    beta_c = phys_params.beta_c
    c_ref = phys_params.c_ref
    e_g = phys_params.e_g
    println("\n>>> Initializing Fully coupled Monolithic MultiField Spaces <<<")

    function res_coupled(x, y)
        u, p, c = x
        v, q, w = y
        
        τ_m_field = τ_m ∘ u
        τ_c_field = τ_c ∘ u
        
        # Buoyancy coupling directly forcing the fluid based on the combined block state of 'c'
        body_force = rho0 * g_mag * beta_c * (c - c_ref) * e_g
        
        # Residual of Navier Stokes (momentum + continuity) + SUPG stabilization
        r_ns = rho0 * (u ⋅ ∇(u)) ⋅ v +
               2 * mu * (ε(u) ⊙ ε(v)) -
               p * (∇ ⋅ v) +
               q * (∇ ⋅ u) -
               body_force ⋅ v +
               τ_m_field * (rho0 * (u ⋅ ∇(v))) ⋅ (rho0 * (u ⋅ ∇(u)) + ∇(p) - body_force)
        
        # Residual of Concentration transport + SUPG mapped directly utilizing the monolithic 'u' state
        r_ad = (u ⋅ ∇(c)) * w +
               D * (∇(c) ⋅ ∇(w)) +
               τ_c_field * (u ⋅ ∇(w)) * (u ⋅ ∇(c))
               
        return ∫( r_ns + r_ad )dΩ
    end

    local op
    @timeit "Setup Monolithic Operator" begin
        op = FEOperator(res_coupled, X_coup, Y_coup)
    end
    newton_opts = config["numerical"]["newton"]
    precond_type = get(newton_opts, "preconditioner_type", "PETSc")
    
    if precond_type == "PETSc"
        println("   > Using Option B: GridapPETSc PCFIELDSPLIT with LSC")
        
        # Explanation of PETSc Options Array:
        # `-pc_type fieldsplit`: Triggers block algebraic decompositions natively inside the C-layer.
        # `-pc_fieldsplit_0_fields 0,2`: Extracts Velocity(0) and Concentration(2) as the top 'A' block.
        # `-pc_fieldsplit_1_fields 1`: Extracts Pressure(1) as the lower 'S' block.
        # `-pc_fieldsplit_type schur`: Formulates the Upper Triangulation Block inverse.
        # `-fieldsplit_1_pc_type lsc`: Estimates the structural zero-bounds using Least Squares Commutator magic.
        petsc_options = "-ksp_type fgmres \
                         -ksp_rtol $(get(config["numerical"]["solver_NS"], "reltol", 1e-6)) \
                         -pc_type fieldsplit \
                         -pc_fieldsplit_0_fields 0,2 \
                         -pc_fieldsplit_1_fields 1 \
                         -pc_fieldsplit_type schur \
                         -pc_fieldsplit_schur_fact_type upper \
                         -pc_fieldsplit_schur_precondition self \
                         -fieldsplit_0_ksp_type preonly \
                         -fieldsplit_0_pc_type gamg \
                         -fieldsplit_1_pc_type lsc \
                         -fieldsplit_1_ksp_type gmres"

        uh_new, ph_new, ch_new = GridapPETSc.with(args=split(petsc_options)) do
            ls = PETScLinearSolver() # Inherits petsc_options string parameters globally to configure factorization
            nls = DampedNewtonSolver(ls; 
                                     tol=newton_opts["tol"], 
                                     max_iters=newton_opts["max_iters"], 
                                     backtrack_factor=newton_opts["backtrack_factor"], 
                                     min_alpha=newton_opts["min_alpha"])
            local xh
            @timeit "Solve Monolithic Nonlinear System" begin
                xh = Gridap.solve(nls, op)
            end
            return xh[1], xh[2], xh[3]
        end
        return uh_new, ph_new, ch_new
        
    elseif precond_type == "GridapSolvers"
        println("   > Using Option A: Native GridapSolvers BlockTriangularPreconditioner with Explicit Mp")
        
        # Analytically constructs a dummy Pressure Mass Matrix (Mp). 
        # This operates uniquely as the Preconditioner 'Block' bounding the 0.0 zeroes in standard Schur Complements.
        a_Mp(p, q) = ∫( q * p )dΩ
        local Mp
        @timeit "Assemble Schur Mass Preconditioner" begin
            Mp = assemble_matrix(a_Mp, Up, Vp)
        end
        
        solver_A = CustomIterativeSolver(:gmres; precond=:amg, reltol=1e-5)
        solver_Mp = CustomIterativeSolver(:cg; precond=:jacobi, reltol=1e-5)
        
        # Links solvers into the Triangulated Block Layout explicitly inside Gridap abstractions
        P = GridapSolvers.LinearSolvers.BlockTriangularSolver([solver_A, solver_Mp])
        ls = GridapSolvers.LinearSolvers.FGMRESSolver(100, P)
        
        nls = DampedNewtonSolver(ls; 
                                 tol=newton_opts["tol"], 
                                 max_iters=newton_opts["max_iters"], 
                                 backtrack_factor=newton_opts["backtrack_factor"], 
                                 min_alpha=newton_opts["min_alpha"])
        local xh
        @timeit "Solve Monolithic Nonlinear System" begin
            xh = Gridap.solve(nls, op)
        end
        return xh[1], xh[2], xh[3]
    else
        # Fallback raw non-preconditioned Krylov evaluation (Likely diverges heavily on finer structures)
        ns_cfg = config["numerical"]["solver_NS"]
        ls = CustomIterativeSolver(Symbol(ns_cfg["type"]); precond=Symbol(ns_cfg["precond"]), reltol=get(ns_cfg, "reltol", 1e-4))
        
        nls = DampedNewtonSolver(ls; 
                                 tol=newton_opts["tol"], 
                                 max_iters=newton_opts["max_iters"], 
                                 backtrack_factor=newton_opts["backtrack_factor"], 
                                 min_alpha=newton_opts["min_alpha"])
                                 
        local xh
        @timeit "Solve Monolithic Nonlinear System" begin
            xh = Gridap.solve(nls, op)
        end
        return xh[1], xh[2], xh[3]
    end
end
