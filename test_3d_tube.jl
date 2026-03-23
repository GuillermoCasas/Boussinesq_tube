using Gridap
using GridapGmsh
using LinearAlgebra

function generate_tube_mesh()
    println("Generating 3D Gmsh mesh using GEO...")
    GridapGmsh.gmsh.initialize()
    GridapGmsh.gmsh.option.setNumber("General.Terminal", 0)
    GridapGmsh.gmsh.model.add("tube")

    L = 5.0
    R = 0.5
    mesh_size = 0.4 # Coarse to keep the test quick

    # Define bottom surface
    GridapGmsh.gmsh.model.geo.addPoint(0, 0, 0, mesh_size, 1)

    GridapGmsh.gmsh.model.geo.addPoint(0, 0, R, mesh_size, 2)
    GridapGmsh.gmsh.model.geo.addPoint(0, R, 0, mesh_size, 3)
    GridapGmsh.gmsh.model.geo.addPoint(0, 0, -R, mesh_size, 4)
    GridapGmsh.gmsh.model.geo.addPoint(0, -R, 0, mesh_size, 5)

    GridapGmsh.gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    GridapGmsh.gmsh.model.geo.addCircleArc(3, 1, 4, 2)
    GridapGmsh.gmsh.model.geo.addCircleArc(4, 1, 5, 3)
    GridapGmsh.gmsh.model.geo.addCircleArc(5, 1, 2, 4)

    GridapGmsh.gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    GridapGmsh.gmsh.model.geo.addPlaneSurface([1], 1)

    # Extrude
    GridapGmsh.gmsh.model.geo.extrude([(2, 1)], L, 0, 0)

    GridapGmsh.gmsh.model.geo.synchronize()

    eps = 1e-3
    inlet_surfs = GridapGmsh.gmsh.model.getEntitiesInBoundingBox(-eps, -R-eps, -R-eps, eps, R+eps, R+eps, 2)
    outlet_surfs = GridapGmsh.gmsh.model.getEntitiesInBoundingBox(L-eps, -R-eps, -R-eps, L+eps, R+eps, R+eps, 2)
    
    all_surfs = GridapGmsh.gmsh.model.getEntities(2)
    wall_surfs = setdiff(all_surfs, inlet_surfs, outlet_surfs)

    GridapGmsh.gmsh.model.addPhysicalGroup(2, [s[2] for s in inlet_surfs], 1)
    GridapGmsh.gmsh.model.setPhysicalName(2, 1, "inlet")

    GridapGmsh.gmsh.model.addPhysicalGroup(2, [s[2] for s in outlet_surfs], 2)
    GridapGmsh.gmsh.model.setPhysicalName(2, 2, "outlet")

    GridapGmsh.gmsh.model.addPhysicalGroup(2, [s[2] for s in wall_surfs], 3)
    GridapGmsh.gmsh.model.setPhysicalName(2, 3, "walls")
    
    vol_tags = [v[2] for v in GridapGmsh.gmsh.model.getEntities(3)]
    GridapGmsh.gmsh.model.addPhysicalGroup(3, vol_tags, 4)
    GridapGmsh.gmsh.model.setPhysicalName(3, 4, "fluid")

    GridapGmsh.gmsh.model.mesh.generate(3)

    GridapGmsh.gmsh.write("data/tube.msh")
    GridapGmsh.gmsh.finalize()
    println("Mesh saved to data/tube.msh")
end

function run_stationary_boussinesq_3d()
    println("\nLoading the stationary Boussinesq model in 3D...")

    L = 5.0
    R = 0.5

    rho0 = 1.0
    mu = 0.05       # Larger mu for stability on coarse grid
    D = 0.01        
    g_mag = 9.81
    beta_c = 0.1
    c_ref = 0.0
    e_g = VectorValue(0.0, 0.0, -1.0) # Gravity in -z

    U_max = 1.0

    c_top = 1.0
    c_mid = 2.0
    c_bot = 3.0

    tol = 1e-4
    max_iters = 30

    # 1. READ MESH
    model = DiscreteModelFromFile("data/tube.msh")

    # The tags are "inlet", "outlet", "walls", "fluid" from Gmsh
    labels = get_face_labeling(model)

    # 2. FE SPACES
    degree_u = 2
    degree_p = 1
    degree_c = 2

    # VectorValue{3,Float64} for 3D!
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, degree_u),
                     conformity=:H1, dirichlet_tags=["inlet", "walls"])
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_p),
                     conformity=:H1) 
    Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_c),
                     conformity=:H1, dirichlet_tags=["inlet"])

    # Inlet profile (paraboloid along X-axis: 0 <= r <= R, flows in X)
    function u_in(x)
        y = x[2]
        z = x[3]
        r2 = y^2 + z^2
        val = U_max * max(0.0, 1.0 - r2/(R^2))
        return VectorValue(val, 0.0, 0.0)
    end
    u_wall(x) = VectorValue(0.0, 0.0, 0.0)

    Uu = TrialFESpace(Vu, [u_in, u_wall])
    Up = TrialFESpace(Vp)

    # Concentration profile layered in Z direction
    function c_in_func(x)
        z = x[3]
        if z < -R/3
            return c_bot
        elseif z < R/3
            return c_mid
        else
            return c_top
        end
    end

    Uc = TrialFESpace(Vc, [c_in_func])

    Y_NS = MultiFieldFESpace([Vu, Vp])
    X_NS = MultiFieldFESpace([Uu, Up])

    # 3. WEAK FORMULATION
    degree_quad = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree_quad)

    # Coarse estimate for h
    h_val = 0.25

    norm_u(u) = sqrt(u ⋅ u) + 1e-12
    τ_m(u) = 1.0 / sqrt( (2.0 * rho0 * norm_u(u) / h_val)^2 + (4.0 * mu / h_val^2)^2 )
    τ_c(u) = 1.0 / sqrt( (2.0 * norm_u(u) / h_val)^2 + (4.0 * D / h_val^2)^2 )

    # 4. PICARD LOOP
    uh_prev = interpolate_everywhere(VectorValue(0.0, 0.0, 0.0), Uu)
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
        uh_new, ph_new = solve(op_NS)
        
        τ_c_field = τ_c ∘ uh_new
        
        a_AD(c, w) = ∫( 
            (uh_new ⋅ ∇(c)) * w + 
            D * (∇(c) ⋅ ∇(w)) + 
            τ_c_field * (uh_new ⋅ ∇(w)) * (uh_new ⋅ ∇(c))
        )dΩ
        
        l_AD(w) = ∫( 0.0 * w )dΩ
        
        op_AD = AffineFEOperator(a_AD, l_AD, Uc, Vc)
        ch_new = solve(op_AD)
        
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

    out_file = "results/tube_boussinesq_3d"
    writevtk(Ω, out_file, cellfields=["velocity"=>uh_new, "pressure"=>ph_new, "concentration"=>ch_new])
    println("Results exported to $out_file.vtu")
end

generate_tube_mesh()
run_stationary_boussinesq_3d()
