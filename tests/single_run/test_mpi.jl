using JSON
using Gridap
using GridapGmsh
using GridapDistributed
using PartitionedArrays
import MPI

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

function run_mpi_test(distribute, np)
    parts = distribute(LinearIndices((np,)))
    
    base_config = JSON.parsefile("../../data/case_options.json")
    test_config = JSON.parsefile("data/test_options.json")
    config = recursive_merge(base_config, test_config)
    
    h = config["test_options"]["mesh_size"]
    config["geometry"]["mesh_size"] = h
    
    msh_path = "meshes/tube_mpi.msh"
    
    # We only want to generate the mesh once from the master thread physically.
    # In distributed environments, usually the mesh is already built, but here we enforce generation safely.
    if i_am_main(parts)
        println("Rank 1 generating mesh...")
        generate_tube_mesh(config, msh_path)
    end
    MPI.Barrier(MPI.COMM_WORLD) 

    # Parts is natively forwarded into discrete load splitting mathematically
    uh, ph, ch, Ω, dΩ = run_simulation(config, msh_path; out_vtk="results/tube_mpi", parts=parts)
    
    # Print the native timer precisely purely avoiding overlapping strings!
    if i_am_main(parts)
        export_timer_summary("results/timing_tube_mpi.txt", "Timing Summary (MPI Distributed: $np cores)")
        print_timer_summary()
        println("MPI Distributed run test completed successfully over $np cores.")
    end
end

with_mpi() do distribute
    # You can change the number of cores dynamically via `mpiexec -n 4 julia test_mpi.jl`
    np = MPI.Comm_size(MPI.COMM_WORLD)
    run_mpi_test(distribute, np)
end
