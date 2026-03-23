using Gridap
using GridapGmsh
using LinearAlgebra
using JSON

include("custom_solvers.jl")

function solve_tube(h_val::Float64, out_name::String, config::Dict)
    println("\n===========================================")
    println("Solving for mesh size h = $h_val")
    println("===========================================")
    
    GridapGmsh.gmsh.initialize()
    GridapGmsh.gmsh.option.setNumber("General.Terminal", 0)
    GridapGmsh.gmsh.model.add("tube")

    L = config["geometry"]["L"]
    R = config["geometry"]["R"]

    GridapGmsh.gmsh.model.geo.addPoint(0, 0, 0, h_val, 1)
    GridapGmsh.gmsh.model.geo.addPoint(0, 0, R, h_val, 2)
    GridapGmsh.gmsh.model.geo.addPoint(0, R, 0, h_val, 3)
    GridapGmsh.gmsh.model.geo.addPoint(0, 0, -R, h_val, 4)
    GridapGmsh.gmsh.model.geo.addPoint(0, -R, 0, h_val, 5)

    GridapGmsh.gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    GridapGmsh.gmsh.model.geo.addCircleArc(3, 1, 4, 2)
    GridapGmsh.gmsh.model.geo.addCircleArc(4, 1, 5, 3)
    GridapGmsh.gmsh.model.geo.addCircleArc(5, 1, 2, 4)

    GridapGmsh.gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    GridapGmsh.gmsh.model.geo.addPlaneSurface([1], 1)

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
    msh_file = "meshes/$out_name.msh"
    GridapGmsh.gmsh.write(msh_file)
    GridapGmsh.gmsh.finalize()

    rho0 = config["physics"]["rho0"]
    mu = config["physics"]["mu"]
    D = config["physics"]["D"]
    g_mag = config["physics"]["g_mag"]
    beta_c = config["physics"]["beta_c"]
    c_ref = config["physics"]["c_ref"]
    e_g = VectorValue(0.0, 0.0, -1.0) 

    U_max = config["boundary_conditions"]["U_max"]
    c_top = config["boundary_conditions"]["c_top"]
    c_mid = config["boundary_conditions"]["c_mid"]
    c_bot = config["boundary_conditions"]["c_bot"]
    tol = config["numerical"]["tol"]
    max_iters = config["numerical"]["max_iters"]

    model = DiscreteModelFromFile(msh_file)
    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, 2), conformity=:H1, dirichlet_tags=["inlet", "walls"])
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1) 
    Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, 2), conformity=:H1, dirichlet_tags=["inlet"])

    u_in(x) = VectorValue(U_max * max(0.0, 1.0 - (x[2]^2 + x[3]^2)/(R^2)), 0.0, 0.0)
    u_wall(x) = VectorValue(0.0, 0.0, 0.0)
    Uu = TrialFESpace(Vu, [u_in, u_wall])
    Up = TrialFESpace(Vp)

    c_in_func(x) = x[3] < -R/3 ? c_bot : (x[3] < R/3 ? c_mid : c_top)
    Uc = TrialFESpace(Vc, [c_in_func])

    Y_NS = MultiFieldFESpace([Vu, Vp])
    X_NS = MultiFieldFESpace([Uu, Up])

    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)

    h_estim = h_val
    norm_u(u) = sqrt(u ⋅ u) + 1e-12
    τ_m(u) = 1.0 / sqrt( (2.0 * rho0 * norm_u(u) / h_estim)^2 + (4.0 * mu / h_estim^2)^2 )
    τ_c(u) = 1.0 / sqrt( (2.0 * norm_u(u) / h_estim)^2 + (4.0 * D / h_estim^2)^2 )

    uh_prev = interpolate_everywhere(VectorValue(0.0, 0.0, 0.0), Uu)
    ph_prev = interpolate_everywhere(0.0, Up)
    ch_prev = interpolate_everywhere(c_in_func, Uc)

    uh_new = uh_prev; ph_new = ph_prev; ch_new = ch_prev

    for iter in 1:max_iters
        τ_m_field = τ_m ∘ uh_prev
        body_force = rho0 * g_mag * beta_c * (ch_prev - c_ref) * e_g
        
        a_NS(X, Y) = ∫( 
            rho0 * (uh_prev ⋅ ∇(X[1])) ⋅ Y[1] + 2 * mu * (ε(X[1]) ⊙ ε(Y[1])) - 
            X[2] * (∇ ⋅ Y[1]) + Y[2] * (∇ ⋅ X[1]) + 
            τ_m_field * (rho0 * (uh_prev ⋅ ∇(Y[1]))) ⋅ (rho0 * (uh_prev ⋅ ∇(X[1])) + ∇(X[2]))
        )dΩ
        l_NS(Y) = ∫( body_force ⋅ Y[1] + τ_m_field * (rho0 * (uh_prev ⋅ ∇(Y[1]))) ⋅ body_force )dΩ
        
        op_NS = AffineFEOperator(a_NS, l_NS, X_NS, Y_NS)
        ns_cfg = config["numerical"]["solver_NS"]
        solver_NS = CustomIterativeSolver(Symbol(ns_cfg["type"]); precond=Symbol(ns_cfg["precond"]), reltol=ns_cfg["reltol"])
        uh_new, ph_new = Gridap.solve(solver_NS, op_NS)
        
        τ_c_field = τ_c ∘ uh_new
        a_AD(c, w) = ∫( (uh_new ⋅ ∇(c)) * w + D * (∇(c) ⋅ ∇(w)) + τ_c_field * (uh_new ⋅ ∇(w)) * (uh_new ⋅ ∇(c)) )dΩ
        l_AD(w) = ∫( 0.0 * w )dΩ
        
        op_AD = AffineFEOperator(a_AD, l_AD, Uc, Vc)
        ad_cfg = config["numerical"]["solver_AD"]
        solver_AD = CustomIterativeSolver(Symbol(ad_cfg["type"]); precond=Symbol(ad_cfg["precond"]), tau=ad_cfg["tau"], reltol=ad_cfg["reltol"])
        ch_new = Gridap.solve(solver_AD, op_AD)
        
        u_err = norm(get_free_dof_values(uh_new) .- get_free_dof_values(uh_prev)) / (norm(get_free_dof_values(uh_new)) + 1e-10)
        c_err = norm(get_free_dof_values(ch_new) .- get_free_dof_values(ch_prev)) / (norm(get_free_dof_values(ch_new)) + 1e-10)
        
        if u_err < tol && c_err < tol break end
        uh_prev = uh_new; ph_prev = ph_new; ch_prev = ch_new
    end
    
    # Export results for visualization
    vtk_path = "results/$(out_name)"
    println("Exporting $(vtk_path).vtu")
    writevtk(Ω, vtk_path, cellfields=["velocity"=>uh_new, "pressure"=>ph_new, "concentration"=>ch_new])

    return uh_new, ph_new, ch_new
end

function run_convergence()
    h_vals = [0.3, 0.2]
    h_ref = 0.15

    config = JSON.parsefile("data/case_options.json")

    println("Generating Reference Mesh (h_ref = $h_ref)")
    u_ref, p_ref, c_ref = solve_tube(h_ref, "tube_ref", config)

    # Fixed core points to avoid boundary evaluation issues
    pts = Point{3,Float64}[]
    for x in range(0.2, 4.8, length=30)
        for y in range(-0.35, 0.35, length=15)
            for z in range(-0.35, 0.35, length=15)
                if y^2 + z^2 < 0.35^2
                    push!(pts, Point(x,y,z))
                end
            end
        end
    end

    valid_pts = Point{3,Float64}[]
    u_ref_vals = VectorValue{3,Float64}[]
    p_ref_vals = Float64[]
    c_ref_vals = Float64[]

    println("Validating Reference Point Evaluation...")
    for pt in pts
        try
            push!(u_ref_vals, u_ref(pt))
            push!(p_ref_vals, p_ref(pt))
            push!(c_ref_vals, c_ref(pt))
            push!(valid_pts, pt)
        catch
        end
    end
    println("Evaluated $(length(valid_pts)) valid points.")

    errors_u = Float64[]; errors_p = Float64[]; errors_c = Float64[]

    for h in h_vals
        uh, ph, ch = solve_tube(h, "tube_$h", config)
        
        e_u_sum = 0.0; e_p_sum = 0.0; e_c_sum = 0.0
        v_count = 0
        
        for (i, pt) in enumerate(valid_pts)
            try
                uv = uh(pt)
                pv = ph(pt)
                cv = ch(pt)
                
                e_u_sum += norm(uv - u_ref_vals[i])^2
                e_p_sum += (pv - p_ref_vals[i])^2
                e_c_sum += (cv - c_ref_vals[i])^2
                v_count += 1
            catch
            end
        end
        
        e_u = sqrt(e_u_sum / v_count)
        e_p = sqrt(e_p_sum / v_count)
        e_c = sqrt(e_c_sum / v_count)
        
        push!(errors_u, e_u); push!(errors_p, e_p); push!(errors_c, e_c)
        println("=> h = $h | Errs: U=$e_u, P=$e_p, C=$e_c (over $v_count pts)")
    end

    open("results/convergence_data.json", "w") do f
        JSON.print(f, Dict("h_vals"=>h_vals, "errors_u"=>errors_u, "errors_p"=>errors_p, "errors_c"=>errors_c), 4)
    end
    println("Convergence discrete L2 data saved.")
end

run_convergence()
