using Gridap
using GridapGmsh
using GridapSolvers
using GridapPETSc
using LinearAlgebra

# Note: Since run_simulation.jl will be included from scripts in tests/*/
# We assume the working directory is still the project root.
include("linear_solvers/custom_solvers.jl")
include("nonlinear_iterators/damped_newton.jl")
include("nonlinear_iterators/solve_newton.jl")
include("nonlinear_iterators/solve_picard.jl")

function generate_tube_mesh(config::Dict, msh_file::String)
    println("Generating 3D Gmsh mesh using GEO...")
    GridapGmsh.gmsh.initialize()
    GridapGmsh.gmsh.option.setNumber("General.Terminal", 0)
    GridapGmsh.gmsh.model.add("tube")

    L = config["geometry"]["L"]
    R = config["geometry"]["R"]
    mesh_size = config["geometry"]["mesh_size"]

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
    GridapGmsh.gmsh.write(msh_file)
    GridapGmsh.gmsh.finalize()
    println("Mesh saved to $msh_file")
end


function run_simulation(config::Dict, msh_file::String; out_vtk::Union{String, Nothing}=nothing)
    println("\nLoading the stationary Boussinesq model in 3D from $msh_file ...")

    R = config["geometry"]["R"]
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

    model = DiscreteModelFromFile(msh_file)

    degree_u = 2
    degree_p = 1
    degree_c = 2

    Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, degree_u),
                     conformity=:H1, dirichlet_tags=["inlet", "walls"])
    Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_p), conformity=:H1) 
    Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_c),
                     conformity=:H1, dirichlet_tags=["inlet"])

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

    function c_in_func(x)
        z = x[3]
        if z < -R/3 return c_bot
        elseif z < R/3 return c_mid
        else return c_top
        end
    end

    Uc = TrialFESpace(Vc, [c_in_func])

    degree_quad = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree_quad)

    h_val = config["geometry"]["mesh_size"]

    norm_u(u) = sqrt(u ⋅ u + 1e-12)
    τ_m(u) = 1.0 / sqrt( (2.0 * rho0 * norm_u(u) / h_val)^2 + (4.0 * mu / h_val^2)^2 )
    τ_c(u) = 1.0 / sqrt( (2.0 * norm_u(u) / h_val)^2 + (4.0 * D / h_val^2)^2 )

    coupling = get(config["numerical"], "coupling", "picard")

    if coupling == "newton"
        X_coup = MultiFieldFESpace([Uu, Up, Uc])
        Y_coup = MultiFieldFESpace([Vu, Vp, Vc])
        uh_new, ph_new, ch_new = solve_boussinesq_newton(config, X_coup, Y_coup, dΩ, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g, Up, Vp)
    else
        X_NS = MultiFieldFESpace([Uu, Up])
        Y_NS = MultiFieldFESpace([Vu, Vp])
        uh_new, ph_new, ch_new = solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, rho0, mu, D, g_mag, beta_c, c_ref, e_g)
    end

    if out_vtk !== nothing
        writevtk(Ω, out_vtk, cellfields=["velocity"=>uh_new, "pressure"=>ph_new, "concentration"=>ch_new])
        println("Results exported to $out_vtk.vtu")
    end
    
    return uh_new, ph_new, ch_new, Ω, dΩ
end
