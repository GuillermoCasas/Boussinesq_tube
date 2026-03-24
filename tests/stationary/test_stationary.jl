using JSON
using Gridap

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

function run_stationary()
    base_config = JSON.parsefile("../../data/case_options.json")
    test_config = JSON.parsefile("data/test_options.json")
    config = recursive_merge(base_config, test_config)
    
    h = config["test_options"]["mesh_size"]
    config["geometry"]["mesh_size"] = h
    
    msh_path = "meshes/tube_stationary.msh"
    generate_tube_mesh(config, msh_path)
    
    uh, ph, ch, Ω, dΩ = run_simulation(config, msh_path; out_vtk="results/tube_stationary")
    println("Stationary Boussinesq test completed successfully.")
end

run_stationary()
