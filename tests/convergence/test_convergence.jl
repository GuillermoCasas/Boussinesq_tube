using JSON
using Gridap
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

function run_convergence()
    base_config = JSON.parsefile("../../data/case_options.json")
    test_config = JSON.parsefile("data/test_options.json")
    config = recursive_merge(base_config, test_config)
    
    h_vals = config["test_options"]["mesh_sizes"]
    h_ref = config["test_options"]["reference_mesh"]

    mkpath("results")
    mkpath("meshes")

    println("Generating Reference Mesh (h_ref = $h_ref)")
    config["geometry"]["mesh_size"] = h_ref
    ref_msh = "meshes/tube_ref.msh"
    generate_tube_mesh(config, ref_msh)
    
    prof_file = "results/profiling_summary.txt"
    if isfile(prof_file)
        rm(prof_file)
    end
    
    u_ref, p_ref, c_ref, Ω_ref, dΩ_ref = run_simulation(config, ref_msh; out_vtk="results/tube_ref")
    export_timer_summary(prof_file, "Timing Summary (h_ref=$h_ref)")
    reset_timer!()

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
        println("\n===========================================")
        println("Solving for mesh size h = $h")
        println("===========================================")
        
        config["geometry"]["mesh_size"] = h
        msh_path = "meshes/tube_$h.msh"
        generate_tube_mesh(config, msh_path)
        
        uh, ph, ch, Ω, dΩ = run_simulation(config, msh_path; out_vtk="results/tube_$h")
        export_timer_summary(prof_file, "Timing Summary (h=$h)")
        reset_timer!()
        
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
