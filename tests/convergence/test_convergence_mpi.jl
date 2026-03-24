using JSON
using Gridap
using GridapGmsh
using GridapDistributed
using PartitionedArrays
import MPI
using LinearAlgebra

include("../../src/run_simulation.jl")

function recursive_merge(x::Dict, y::Dict)
    z = copy(x)
    for (k, v) in y
        if haskey(z, k) && isa(z[k], Dict) && isa(v, Dict)
            z[k] = recursive_merge(z[k], v)
        else
            z[k] = v
        end
    end
    return z
end

function run_convergence_mpi(distribute, np)
    parts = distribute(LinearIndices((np,)))

    base_config = JSON.parsefile("../../data/case_options.json")
    test_config = JSON.parsefile("data/test_options.json")
    config = recursive_merge(base_config, test_config)
    
    h_vals = config["test_options"]["mesh_sizes"]
    h_ref = config["test_options"]["reference_mesh"]

    if i_am_main(parts)
        mkpath("results")
        mkpath("meshes")
        println("Generating Reference Mesh (h_ref = $h_ref)")
        config["geometry"]["mesh_size"] = h_ref
        generate_tube_mesh(config, "meshes/tube_ref_mpi.msh")
    end
    MPI.Barrier(MPI.COMM_WORLD)

    u_ref, p_ref, c_ref, Ω_ref, dΩ_ref = run_simulation(config, "meshes/tube_ref_mpi.msh"; out_vtk="results/tube_mpi_ref", parts=parts)
    
    if i_am_main(parts)
        export_timer_summary("results/timing_tube_mpi_ref.txt", "Timing Summary (MPI h_ref=$h_ref)")
        reset_timer!()
    end
    
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

    # Map the discrete evaluators explicitly across localized partitions!
    local_valid_dict = Dict{Int, Tuple{VectorValue{3,Float64}, Float64, Float64}}()
    
    map_parts(local_views(u_ref), local_views(p_ref), local_views(c_ref)) do u_l, p_l, c_l
        for (i, pt) in enumerate(pts)
            try
                local_valid_dict[i] = (u_l(pt), p_l(pt), c_l(pt))
            catch
            end
        end
    end

    local_count = length(local_valid_dict)
    global_count = MPI.Allreduce(local_count, MPI.SUM, MPI.COMM_WORLD)
    if i_am_main(parts)
        println("Evaluated $global_count valid discrete point nodes natively across the MPI distribution.")
    end

    errors_u = Float64[]; errors_p = Float64[]; errors_c = Float64[]

    for h in h_vals
        if i_am_main(parts)
            println("\n===========================================")
            println("Solving for mesh size h = $h")
            println("===========================================")
            config["geometry"]["mesh_size"] = h
            generate_tube_mesh(config, "meshes/tube_mpi_$h.msh")
        end
        MPI.Barrier(MPI.COMM_WORLD)

        uh, ph, ch, Ω, dΩ = run_simulation(config, "meshes/tube_mpi_$h.msh"; out_vtk="results/tube_mpi_$h", parts=parts)
        
        if i_am_main(parts)
            export_timer_summary("results/timing_tube_mpi_$(h).txt", "Timing Summary (MPI h=$h)")
            reset_timer!()
        end
        
        e_u_sum_local = Ref(0.0)
        e_p_sum_local = Ref(0.0)
        e_c_sum_local = Ref(0.0)
        v_count_local = Ref(0)
        
        map_parts(local_views(uh), local_views(ph), local_views(ch)) do uh_l, ph_l, ch_l
            for (i, refs) in local_valid_dict
                try
                    uv = uh_l(pts[i])
                    pv = ph_l(pts[i])
                    cv = ch_l(pts[i])
                    
                    e_u_sum_local[] += norm(uv - refs[1])^2
                    e_p_sum_local[] += (pv - refs[2])^2
                    e_c_sum_local[] += (cv - refs[3])^2
                    v_count_local[] += 1
                catch
                end
            end
        end
        
        # Merge physical error bounds independently across CPUs 
        e_u_sum_g = MPI.Allreduce(e_u_sum_local[], MPI.SUM, MPI.COMM_WORLD)
        e_p_sum_g = MPI.Allreduce(e_p_sum_local[], MPI.SUM, MPI.COMM_WORLD)
        e_c_sum_g = MPI.Allreduce(e_c_sum_local[], MPI.SUM, MPI.COMM_WORLD)
        v_count_g = MPI.Allreduce(v_count_local[], MPI.SUM, MPI.COMM_WORLD)
        
        e_u = sqrt(e_u_sum_g / (v_count_g + 1e-10))
        e_p = sqrt(e_p_sum_g / (v_count_g + 1e-10))
        e_c = sqrt(e_c_sum_g / (v_count_g + 1e-10))
        
        if i_am_main(parts)
            push!(errors_u, e_u); push!(errors_p, e_p); push!(errors_c, e_c)
            println("=> h = $h | Errs: U=$e_u, P=$e_p, C=$e_c (over $v_count_g MPI-mapped pts)")
        end
    end

    if i_am_main(parts)
        open("results/convergence_data_mpi.json", "w") do f
            JSON.print(f, Dict("h_vals"=>h_vals, "errors_u"=>errors_u, "errors_p"=>errors_p, "errors_c"=>errors_c), 4)
        end
        println("MPI Convergence discrete L2 data saved successfully.")
    end
end

with_mpi() do distribute
    np = MPI.Comm_size(MPI.COMM_WORLD)
    run_convergence_mpi(distribute, np)
end
