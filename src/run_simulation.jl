using Gridap
using GridapGmsh
using GridapSolvers
using GridapPETSc
using LinearAlgebra

# Note: Since run_simulation.jl will be included from scripts in tests/*/
# We assume the working directory is still the project root.

include("timer_utils.jl")
using .TimerUtils

include("linear_solvers/custom_solvers.jl")
include("nonlinear_iterators/damped_newton.jl")
include("nonlinear_iterators/solve_newton.jl")
include("nonlinear_iterators/solve_picard.jl")
include("nonlinear_iterators/solve_staggered_newton.jl")

struct PhysicalParameters
    rho0::Float64
    mu::Float64
    D::Float64
    g_mag::Float64
    beta_c::Float64
    c_ref::Float64
    e_g::VectorValue{3, Float64}
end

function generate_tube_mesh(config::Dict, msh_file::String)
    println("Generating 3D Gmsh mesh using GEO...")
    @timeit "Mesh Generation" begin
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
    end
    println("Mesh saved to $msh_file")
end


function run_simulation(config::Dict, msh_file::String; out_vtk::Union{String, Nothing}=nothing, parts=nothing)
    println("\nLoading the stationary Boussinesq model in 3D from $msh_file ...")

    R = config["geometry"]["R"]
    rho0 = config["physics"]["rho0"]
    mu = config["physics"]["mu"]       
    D = config["physics"]["D"]        
    g_mag = config["physics"]["g_mag"]
    beta_c = config["physics"]["beta_c"]
    c_ref = config["physics"]["c_ref"]
    e_g = VectorValue(0.0, 0.0, -1.0)
    
    phys_params = PhysicalParameters(rho0, mu, D, g_mag, beta_c, c_ref, e_g)

    U_max = config["boundary_conditions"]["U_max"]
    c_top = config["boundary_conditions"]["c_top"]
    c_mid = config["boundary_conditions"]["c_mid"]
    c_bot = config["boundary_conditions"]["c_bot"]

    local model
    @timeit "Model Loading" begin
        if parts === nothing
            model = DiscreteModelFromFile(msh_file)
        else
            model = GmshDiscreteModel(parts, msh_file)
        end
    end

    degree_u = 2
    degree_p = 1
    degree_c = 2

    local Vu, Vp, Vc, Uu, Up, Uc
    @timeit "FESpace Allocation" begin
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
    end

    degree_quad = 4
    local Ω, dΩ
    @timeit "Triangulation Mapping" begin
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree_quad)
    end

    h_val = config["geometry"]["mesh_size"]

    norm_u(u) = sqrt(u ⋅ u + 1e-12)
    τ_m(u) = 1.0 / sqrt( (2.0 * rho0 * norm_u(u) / h_val)^2 + (4.0 * mu / h_val^2)^2 )
    τ_c(u) = 1.0 / sqrt( (2.0 * norm_u(u) / h_val)^2 + (4.0 * D / h_val^2)^2 )

    strategy = get(config["numerical"], "nonlinear_strategy", "staggered")
    ns_cfg = config["numerical"]["solver_NS"]

    function _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, strategy)
        if strategy == "monolithic"
            println("\n=======================================================")
            println("=> SOLVER ARCHITECTURE: MONOLITHIC (Damped Newton)")
            println("=======================================================\n")
            X_coup = MultiFieldFESpace([Uu, Up, Uc])
            Y_coup = MultiFieldFESpace([Vu, Vp, Vc])
            return solve_boussinesq_newton(config, X_coup, Y_coup, dΩ, τ_m, τ_c, phys_params, Up, Vp)
        elseif strategy == "staggered_newton"
            println("\n=======================================================")
            println("=> SOLVER ARCHITECTURE: STAGGERED NEWTON (Decoupled NS Jacobian)")
            println("=======================================================\n")
            return solve_boussinesq_staggered_newton(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params)
        else
            println("\n=======================================================")
            println("=> SOLVER ARCHITECTURE: STAGGERED (Decoupled Picard)")
            println("=======================================================\n")
            return solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params)
        end
    end

    local uh_new, ph_new, ch_new
    if get(ns_cfg, "type", "") == "petsc_amg"
        X_NS = MultiFieldFESpace([Uu, Up])
        Y_NS = MultiFieldFESpace([Vu, Vp])
        
        petsc_opts = "-ksp_type fgmres -ksp_rtol 1e-4 -ksp_monitor " *
                     "-mat_block_size 4 " *
                     "-pc_type fieldsplit -pc_fieldsplit_type schur " *
                     "-pc_fieldsplit_schur_fact_type upper " *
                     "-pc_fieldsplit_schur_precondition self " *
                     "-pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 " *
                     "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type gamg " *
                     "-fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type lsc"
                     
        uh_new, ph_new, ch_new = GridapPETSc.with(args=split(petsc_opts)) do
            _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, strategy)
        end
    elseif get(ns_cfg, "type", "") == "petsc"
        X_NS = MultiFieldFESpace([Uu, Up])
        Y_NS = MultiFieldFESpace([Vu, Vp])
        
        petsc_opts = "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
        uh_new, ph_new, ch_new = GridapPETSc.with(args=split(petsc_opts)) do
            _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, strategy)
        end
    elseif get(ns_cfg, "type", "") == "GridapSolvers"
        X_NS = MultiFieldFESpace([Uu, Up]; style=Gridap.MultiField.BlockMultiFieldStyle())
        Y_NS = MultiFieldFESpace([Vu, Vp]; style=Gridap.MultiField.BlockMultiFieldStyle())
        uh_new, ph_new, ch_new = _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, strategy)
    else
        X_NS = MultiFieldFESpace([Uu, Up])
        Y_NS = MultiFieldFESpace([Vu, Vp])
        uh_new, ph_new, ch_new = _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func, τ_m, τ_c, phys_params, strategy)
    end

    @timeit "VTK IO Dump" begin
        if out_vtk !== nothing
            writevtk(Ω, out_vtk, cellfields=["velocity"=>uh_new, "pressure"=>ph_new, "concentration"=>ch_new])
            println("Results exported to $out_vtk.vtu")
        end
    end
    
    return uh_new, ph_new, ch_new, Ω, dΩ
end
