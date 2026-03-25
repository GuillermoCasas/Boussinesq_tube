using Gridap
using GridapGmsh
using GridapSolvers
using GridapPETSc
using LinearAlgebra

# Note: Since run_simulation.jl will be included from scripts in tests/*/
# We assume the working directory is still the project root.

include("timer_utils.jl")
using .TimerUtils

include("nonlinear_iterators/damped_newton.jl")
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


function run_simulation(config::Dict, msh_file::Union{String, Nothing}=nothing; 
                        out_vtk::Union{String, Nothing}=nothing, 
                        parts=nothing, 
                        is_mms::Bool=false,
                        u_exact=nothing, force_u=nothing,
                        c_exact=nothing, force_c=nothing)

    println("\nLoading the Boussinesq model (is_mms=$is_mms)...")

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
    h_val = config["geometry"]["mesh_size"]

    local model
    @timeit "Model Loading" begin
        if is_mms
            L_mms = get(config["geometry"], "L", 1.0)
            domain = (0.0, L_mms, 0.0, L_mms, 0.0, L_mms)
            cells = (Int(round(L_mms/h_val)), Int(round(L_mms/h_val)), Int(round(L_mms/h_val)))
            if parts === nothing
                model = CartesianDiscreteModel(domain, cells)
            else
                model = CartesianDiscreteModel(parts, domain, cells)
            end
        else
            if parts === nothing
                model = DiscreteModelFromFile(msh_file)
            else
                model = GmshDiscreteModel(parts, msh_file)
            end
        end
    end

    degree_u = 1 # Drop to strictly equal-order P1/P1 elements to natively sidestep the disparate ghost layouts over MPI!
    degree_p = 1
    degree_c = 2

    local Vu, Vp, Vc, Uu, Up, Uc
    @timeit "FESpace Allocation" begin
        Vu = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{3,Float64}, degree_u),
                         conformity=:H1, dirichlet_tags=(is_mms ? "boundary" : ["inlet", "walls"]))
        Vp = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_p), conformity=:H1) 
        Vc = TestFESpace(model, ReferenceFE(lagrangian, Float64, degree_c),
                         conformity=:H1, dirichlet_tags=(is_mms ? "boundary" : ["inlet"]))

        if is_mms
            Uu = TrialFESpace(Vu, u_exact)
            Up = TrialFESpace(Vp)
            Uc = TrialFESpace(Vc, c_exact)
        else
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
    end

    degree_quad = 4
    local Ω, dΩ
    @timeit "Triangulation Mapping" begin
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree_quad)
    end

    norm_u(u) = sqrt(u ⋅ u + 1e-12)
    τ_m(u) = 1.0 / sqrt( (2.0 * rho0 * norm_u(u) / h_val)^2 + (4.0 * mu / h_val^2)^2 )
    τ_c(u) = 1.0 / sqrt( (2.0 * norm_u(u) / h_val)^2 + (4.0 * D / h_val^2)^2 )

    strategy = get(config["numerical"], "nonlinear_strategy", "staggered")
    ns_cfg = config["numerical"]["solver_NS"]

    c_in_pass = is_mms ? c_exact : c_in_func

    function _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func_pass, τ_m, τ_c, phys_params, strategy)
        if strategy == "staggered_newton"
            println("\n=======================================================")
            println("=> SOLVER ARCHITECTURE: STAGGERED NEWTON (Decoupled NS Jacobian)")
            println("=======================================================\n")
            return solve_boussinesq_staggered_newton(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func_pass, τ_m, τ_c, phys_params, force_u, force_c)
        else
            println("\n=======================================================")
            println("=> SOLVER ARCHITECTURE: STAGGERED (Decoupled Picard)")
            println("=======================================================\n")
            return solve_boussinesq_picard(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_func_pass, τ_m, τ_c, phys_params, force_u, force_c)
        end
    end

    local uh_new, ph_new, ch_new
    if get(ns_cfg, "type", "") == "petsc_amg"
        # Equal-order topology safely aligns the Block MPI layout dynamically preventing boundary exceptions identically elegantly
        X_NS = MultiFieldFESpace([Uu, Up]; style=Gridap.MultiField.BlockMultiFieldStyle())
        Y_NS = MultiFieldFESpace([Vu, Vp]; style=Gridap.MultiField.BlockMultiFieldStyle())
        
        petsc_opts = "-ns_ksp_type fgmres -ns_ksp_rtol 1e-4 -ns_ksp_monitor " *
                     "-ns_pc_type hypre -ns_pc_hypre_type boomeramg -ns_pc_hypre_boomeramg_max_iter 5"
                     
        uh_new, ph_new, ch_new = GridapPETSc.with(args=String.(split(petsc_opts))) do
            _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_pass, τ_m, τ_c, phys_params, strategy)
        end
    elseif get(ns_cfg, "type", "") == "petsc"
        X_NS = MultiFieldFESpace([Uu, Up])
        Y_NS = MultiFieldFESpace([Vu, Vp])
        
        petsc_opts = "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
        uh_new, ph_new, ch_new = GridapPETSc.with(args=split(petsc_opts)) do
            _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_pass, τ_m, τ_c, phys_params, strategy)
        end
    elseif get(ns_cfg, "type", "") == "GridapSolvers"
        X_NS = MultiFieldFESpace([Uu, Up]; style=Gridap.MultiField.BlockMultiFieldStyle())
        Y_NS = MultiFieldFESpace([Vu, Vp]; style=Gridap.MultiField.BlockMultiFieldStyle())
        # Provide exactly explicit global PETSc context securely mapping BoomerAMG dynamically explicitly inside nested blocks over MPI.
        petsc_opts = "-u_ksp_type gmres -u_pc_type hypre -u_pc_hypre_type boomeramg -u_pc_hypre_boomeramg_max_iter 2 -u_pc_hypre_boomeramg_tol 0.0 " *
                     "-p_ksp_type cg -p_ksp_constant_null_space -p_pc_type hypre -p_pc_hypre_type boomeramg -p_pc_hypre_boomeramg_max_iter 2 -p_pc_hypre_boomeramg_tol 0.0"

        # Fallback if Hypre is NOT available (fallback to GAMG, never Jacobi):
        # petsc_opts = "-u_ksp_type gmres -u_pc_type gamg -u_pc_gamg_type agg -u_pc_gamg_sym_graph true " *
        #              "-p_ksp_type cg -p_ksp_constant_null_space -p_pc_type gamg -p_pc_gamg_type agg -p_pc_gamg_sym_graph true"
                     
        uh_new, ph_new, ch_new = GridapPETSc.with(args=split(petsc_opts)) do
            _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_pass, τ_m, τ_c, phys_params, strategy)
        end
    else
        X_NS = MultiFieldFESpace([Uu, Up])
        Y_NS = MultiFieldFESpace([Vu, Vp])
        uh_new, ph_new, ch_new = _execute_solve(config, X_NS, Y_NS, Uc, Vc, dΩ, Uu, Up, c_in_pass, τ_m, τ_c, phys_params, strategy)
    end

    @timeit "VTK IO Dump" begin
        if out_vtk !== nothing
            writevtk(Ω, out_vtk, cellfields=["velocity"=>uh_new, "pressure"=>ph_new, "concentration"=>ch_new])
            println("Results exported to $out_vtk.vtu")
        end
    end
    
    return uh_new, ph_new, ch_new, Ω, dΩ
end
