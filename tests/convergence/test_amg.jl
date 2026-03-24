using JSON
using Gridap

include("../../src/run_simulation.jl")

function test_multigrid()
    config = JSON.parsefile("../../data/case_options.json")
    
    # We force the exact Custom Iterative solver natively wrapping AMG Schur
    config["numerical"]["solver_NS"]["type"] = "gmres"
    config["numerical"]["solver_NS"]["precond"] = "schur"
    
    # Fast mesh for testing
    h = 0.4
    config["geometry"]["mesh_size"] = h
    msh_path = "meshes/tube_test.msh"
    mkpath("meshes")
    generate_tube_mesh(config, msh_path)
    
    uh, ph, ch, Ω, dΩ = run_simulation(config, msh_path)
    println("Test completed successfully!")
end

test_multigrid()
