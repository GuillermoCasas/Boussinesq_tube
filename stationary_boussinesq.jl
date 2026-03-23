using Gridap
using LinearAlgebra

function run_stationary_boussinesq()
    println("Setting up the stationary Boussinesq model...")

    # ==============================================================================
    # 1. PARAMETERS
    # ==============================================================================
    # Geometry
    L = 5.0  # Length of the tube
    H = 1.0  # Height of the tube
    nx = 50  # Mesh resolution in x
    ny = 15  # Mesh resolution in y

    # Physical parameters
    rho0 = 1.0        # Reference density
    mu = 0.01         # Dynamic viscosity
    D = 0.005         # Solute diffusivity
    g_mag = 9.81      # Gravitational acceleration magnitude
    beta_c = 0.1      # Solutal expansion coefficient
    c_ref = 0.0       # Reference concentration
    e_g = VectorValue(0.0, -1.0) # Gravity direction (acts downwards strictly in y)

    # Boundary conditions
    U_max = 1.0       # Maximum inlet velocity
    c1 = 1.0          # Concentration bottom layer [0, H/3)
    c2 = 0.0          # Concentration middle layer [H/3, 2H/3)
    c3 = 1.0          # Concentration top layer    [2H/3, H]

    # Numerical parameters for coupled solve
    tol = 1e-6
    max_iters = 50

    # ==============================================================================
    # 2. MESH AND BOUNDARY TAGS
    # ==============================================================================
    domain = (0.0, L, 0.0, H)
    partition = (nx, ny)
    model = CartesianDiscreteModel(domain, partition)

    labels = get_face_labeling(model)

    # The default CartesianDiscreteModel tags:
    # 1..4: corners, 5: y=0, 6: y=H, 7: x=0, 8: x=L
    # Explicitly separate the physical boundaries mathematically:
    add_tag_from_tags!(labels, "inlet_u", [7])
    add_tag_from_tags!(labels, "walls_u", [1, 2, 3, 4, 5, 6])
    add_tag_from_tags!(labels, "inlet_c", [1, 3, 7])

    # ==============================================================================
    # 3. FE SPACES
    # ==============================================================================
    # Velocity-Pressure: Taylor-Hood elements (Q2/Q1) to satisfy inf-sup intrinsically
    # Concentration: Q2, compatible with the velocity space for accuracy
    degree_u = 2
    degree_p = 1
    degree_c = 2

    # Weak outflow is naturally imposed on the outlet (x=L) since there's no Dirichlet
    # condition there, setting the fluid traction and the diffusive concentration flux to 0.
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, degree_u),
                     conformity=:H1, dirichlet_tags=["inlet_u", "walls_u"])
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_p),
                     conformity=:H1) 
    Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_c),
                     conformity=:H1, dirichlet_tags=["inlet_c"])

    # Inlet velocity: Parabolic profile perfectly matching 0.0 at y=0, y=H
    u_in(x) = VectorValue(4 * U_max * x[2] * (H - x[2]) / (H^2), 0.0)
    u_wall(x) = VectorValue(0.0, 0.0)

    Uu = TrialFESpace(Vu, [u_in, u_wall])
    Up = TrialFESpace(Vp)

    # Inlet concentration profile via a robust piecewise spatial check
    function c_in(x)
        y = x[2]
        if y < H/3
            return c1
        elseif y < 2*H/3
            return c2
        else
            return c3
        end
    end

    Uc = TrialFESpace(Vc, [c_in])

    # Cartesian product spaces for fully-coupled operators if required, 
    # but we use multi-field constructs for the components.
    Y_NS = MultiFieldFESpace([Vu, Vp])
    X_NS = MultiFieldFESpace([Uu, Up])

    # ==============================================================================
    # 4. WEAK FORMULATION AND STABILIZATION SETUP
    # ==============================================================================
    degree_quad = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree_quad)

    # Measure of mesh cell size for stability terms
    hx = L / nx
    hy = H / ny
    h_val = sqrt(hx^2 + hy^2)

    # VMS / SUPG stabilization factors: Using asymptotic formulas
    norm_u(u) = sqrt(u ⋅ u) + 1e-12
    τ_m(u) = 1.0 / sqrt( (2.0 * rho0 * norm_u(u) / h_val)^2 + (4.0 * mu / h_val^2)^2 )
    τ_c(u) = 1.0 / sqrt( (2.0 * norm_u(u) / h_val)^2 + (4.0 * D / h_val^2)^2 )

    # ==============================================================================
    # 5. COUPLED FIXED-POINT (PICARD) ITERATION
    # ==============================================================================
    println("\nStarting coupled stationary solve using Picard iteration strategy...")

    # Initialization of functional iteration states
    uh_prev = interpolate_everywhere(VectorValue(0.0, 0.0), Uu)
    ph_prev = interpolate_everywhere(0.0, Up)
    ch_prev = interpolate_everywhere(c_in, Uc)

    uh_new = uh_prev
    ph_new = ph_prev
    ch_new = ch_prev

    for iter in 1:max_iters
        println("--- Iteration $iter ---")
        
        # ----------------------------------------------------------------------
        # Step A: Flow solve (Linearized Navier-Stokes driven by previous C)
        # ----------------------------------------------------------------------
        τ_m_field = τ_m ∘ uh_prev
        body_force = rho0 * g_mag * beta_c * (ch_prev - c_ref) * e_g
        
        # Stabilized stationary Oseen-like form (Picard linearization):
        # We explicitly apply SUPG only to the convective and pressure gradients,
        # making standard assumptions of affine mapping where viscous residual ~ 0.
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
        uh_new, ph_new = solve(op_NS)
        
        # ----------------------------------------------------------------------
        # Step B: Solute transport solve (Advection-Diffusion driven by new U)
        # ----------------------------------------------------------------------
        τ_c_field = τ_c ∘ uh_new
        
        # Stabilized stationary Convection-Diffusion weak form:
        a_AD(c, w) = ∫( 
            (uh_new ⋅ ∇(c)) * w + 
            D * (∇(c) ⋅ ∇(w)) + 
            τ_c_field * (uh_new ⋅ ∇(w)) * (uh_new ⋅ ∇(c))
        )dΩ
        
        l_AD(w) = ∫( 0.0 * w )dΩ
        
        op_AD = AffineFEOperator(a_AD, l_AD, Uc, Vc)
        ch_new = solve(op_AD)
        
        # ----------------------------------------------------------------------
        # Step C: Convergence Verification
        # ----------------------------------------------------------------------
        du_norm = norm(get_free_dof_values(uh_new) .- get_free_dof_values(uh_prev))
        dc_norm = norm(get_free_dof_values(ch_new) .- get_free_dof_values(ch_prev))
        
        u_norm_tot = norm(get_free_dof_values(uh_new)) + 1e-10
        c_norm_tot = norm(get_free_dof_values(ch_new)) + 1e-10
        
        err_u = du_norm / u_norm_tot
        err_c = dc_norm / c_norm_tot
        
        println(" Relative error U: $(round(err_u, sigdigits=4))")
        println(" Relative error C: $(round(err_c, sigdigits=4))")
        
        if err_u < tol && err_c < tol
            println("\nCoupled steady-state solution converged in $iter iterations!\n")
            break
        end
        
        uh_prev = uh_new
        ph_prev = ph_new
        ch_prev = ch_new
        
        if iter == max_iters
            println("\nWarning: Maximum iterations reached without full convergence.\n")
        end
    end

    # ==============================================================================
    # 6. EXPORT RESULTS
    # ==============================================================================
    out_file = "stationary_boussinesq"
    writevtk(Ω, out_file, cellfields=["velocity"=>uh_new, "pressure"=>ph_new, "concentration"=>ch_new])
    println("Results exported to $out_file.vtu for visualization in ParaView.")
    
    return uh_new, ph_new, ch_new
end

run_stationary_boussinesq()
