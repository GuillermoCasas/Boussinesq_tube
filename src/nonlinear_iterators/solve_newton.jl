using Gridap
using GridapSolvers
using GridapPETSc

function solve_boussinesq_newton(config, X_coup, Y_coup, dΩ, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g, Up=nothing, Vp=nothing)
    println("\n>>> Initializing Fully coupled Monolithic MultiField Spaces <<<")

    function res_coupled(x, y)
        u, p, c = x
        v, q, w = y
        
        τ_m_field = τ_m ∘ u
        τ_c_field = τ_c ∘ u
        
        body_force = rho0 * g_mag * beta_c * (c - c_ref) * e_g
        
        r_ns = rho0 * (u ⋅ ∇(u)) ⋅ v +
               2 * mu * (ε(u) ⊙ ε(v)) -
               p * (∇ ⋅ v) +
               q * (∇ ⋅ u) -
               body_force ⋅ v +
               τ_m_field * (rho0 * (u ⋅ ∇(v))) ⋅ (rho0 * (u ⋅ ∇(u)) + ∇(p) - body_force)
        
        r_ad = (u ⋅ ∇(c)) * w +
               D * (∇(c) ⋅ ∇(w)) +
               τ_c_field * (u ⋅ ∇(w)) * (u ⋅ ∇(c))
               
        return ∫( r_ns + r_ad )dΩ
    end

    op = FEOperator(res_coupled, X_coup, Y_coup)
    newton_opts = config["numerical"]["newton"]
    precond_type = get(newton_opts, "preconditioner_type", "PETSc")
    
    if precond_type == "PETSc"
        println("   > Using Option B: GridapPETSc PCFIELDSPLIT with LSC")
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
            ls = PETScLinearSolver()
            nls = DampedNewtonSolver(ls; 
                                     tol=newton_opts["tol"], 
                                     max_iters=newton_opts["max_iters"], 
                                     backtrack_factor=newton_opts["backtrack_factor"], 
                                     min_alpha=newton_opts["min_alpha"])
            xh = Gridap.solve(nls, op)
            return xh[1], xh[2], xh[3]
        end
        return uh_new, ph_new, ch_new
        
    elseif precond_type == "GridapSolvers"
        println("   > Using Option A: Native GridapSolvers BlockTriangularPreconditioner with Explicit Mp")
        a_Mp(p, q) = ∫( q * p )dΩ
        Mp = assemble_matrix(a_Mp, Up, Vp)
        
        solver_A = CustomIterativeSolver(:gmres; precond=:amg, reltol=1e-5)
        solver_Mp = CustomIterativeSolver(:cg; precond=:jacobi, reltol=1e-5)
        
        P = GridapSolvers.LinearSolvers.BlockTriangularSolver([solver_A, solver_Mp])
        ls = GridapSolvers.LinearSolvers.FGMRESSolver(100, P)
        
        nls = DampedNewtonSolver(ls; 
                                 tol=newton_opts["tol"], 
                                 max_iters=newton_opts["max_iters"], 
                                 backtrack_factor=newton_opts["backtrack_factor"], 
                                 min_alpha=newton_opts["min_alpha"])
        xh = Gridap.solve(nls, op)
        return xh[1], xh[2], xh[3]
    else
        ns_cfg = config["numerical"]["solver_NS"]
        ls = CustomIterativeSolver(Symbol(ns_cfg["type"]); precond=Symbol(ns_cfg["precond"]), reltol=get(ns_cfg, "reltol", 1e-4))
        
        nls = DampedNewtonSolver(ls; 
                                 tol=newton_opts["tol"], 
                                 max_iters=newton_opts["max_iters"], 
                                 backtrack_factor=newton_opts["backtrack_factor"], 
                                 min_alpha=newton_opts["min_alpha"])
                                 
        xh = Gridap.solve(nls, op)
        return xh[1], xh[2], xh[3]
    end
end
