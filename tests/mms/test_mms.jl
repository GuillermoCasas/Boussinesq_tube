using Gridap
using LinearAlgebra
import JSON

include("../../src/run_simulation.jl")

function run_mms_convergence()
    println("===================================================================")
    println("          METHOD OF MANUFACTURED SOLUTIONS (MMS) CONVERGENCE       ")
    println("===================================================================")

    config_path = joinpath(@__DIR__, "../../data/case_options.json")
    config = JSON.parsefile(config_path)

    # Use pure Picard staggered logic to independently verify boundaries correctly
    config["numerical"]["nonlinear_strategy"] = "staggered"
    
    # CRITICAL: For MMS geometrical O(h^3) Convergence, Linear iterations mathematically inject artificial errors!
    # Lock inner linear solvers exactly to our new fieldsplit block preconditioner
    config["numerical"]["solver_NS"]["type"] = "lu"
    config["numerical"]["solver_AD"]["type"] = "lu"

    # Physics from config
    rho0 = config["physics"]["rho0"]
    mu = config["physics"]["mu"]
    D = config["physics"]["D"]
    
    # -------------------------------------------------------------------------
    # 1. Exact Analytical Solutions
    # -------------------------------------------------------------------------
    u_ex(x) = VectorValue(sin(pi*x[1])*cos(pi*x[2]), -cos(pi*x[1])*sin(pi*x[2]), 0.0)
    
    # ϵ_phys penalty forces the numerical pressure exactly to a mean of zero (∫ p dΩ = 0).
    # Since ∫ sin(πx)sin(πy) dΩ = 4/π^2 ≈ 0.40528, we must shift the analytical solution down appropriately so they precisely match!
    p_ex(x) = sin(pi*x[1])*sin(pi*x[2]) - 4.0/(pi^2)
    
    c_ex(x) = x[1] + x[2] + x[3]

    # -------------------------------------------------------------------------
    # 2. Derive Exact Forcing Terms via ForwardDiff Native in Gridap
    # -------------------------------------------------------------------------
    # Navier-Stokes Momentum Forcing: ρ₀(u⋅∇)u - μΔu + ∇p = f_u
    f_u(x) = rho0 * (∇(u_ex)(x)' ⋅ u_ex(x)) - mu * Δ(u_ex)(x) + ∇(p_ex)(x)

    f_c(x) = (u_ex(x) ⋅ ∇(c_ex)(x)) - D * Δ(c_ex)(x)

    mms_config_path = joinpath(@__DIR__, "data/config.json")
    mms_config = JSON.parsefile(mms_config_path)
    Ns = mms_config["Ns"]
    hs = 1.0 ./ Ns
    
    errors_u = Float64[]
    errors_p = Float64[]
    errors_c = Float64[]

    results_dir = joinpath(@__DIR__, "results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    prof_file = joinpath(results_dir, "profiling_summary.txt")
    if isfile(prof_file)
        rm(prof_file)
    end

    for N in Ns
        h = 1.0 / N
        println("\n=> Running MMS on Mesh N=$N (h=$h)...")
        config["geometry"]["mesh_size"] = h
        config["geometry"]["L"] = 1.0

        out_vtk = joinpath(results_dir, "mms_N_$(N)")
        
        config["numerical"]["solver_NS"]["type"] = "lu"
        uh, ph, ch, Ω, dΩ = run_simulation(config, nothing; 
                                           out_vtk=nothing, 
                                           is_mms=true,
                                           u_exact=u_ex, force_u=f_u,
                                           c_exact=c_ex, force_c=f_c)
        
        export_timer_summary(prof_file, "Timing Summary (h=$h)")
        reset_timer!()

        # -------------------------------------------------------------------------
        # 3. Assess Mathematical L2 Error dynamically over standard quadratures
        # -------------------------------------------------------------------------
        eh_u = uh - u_ex
        eh_p = ph - p_ex
        eh_c = ch - c_ex

        l2_u = sqrt(sum(∫( eh_u ⋅ eh_u )dΩ))
        l2_p = sqrt(sum(∫( eh_p * eh_p )dΩ))
        l2_c = sqrt(sum(∫( eh_c * eh_c )dΩ))

        writevtk(Ω, out_vtk, cellfields=[
            "velocity" => uh,
            "pressure" => ph,
            "concentration" => ch,
            "velocity_exact" => u_ex,
            "pressure_exact" => p_ex,
            "concentration_exact" => c_ex,
            "velocity_error" => eh_u,
            "pressure_error" => eh_p,
            "concentration_error" => eh_c
        ])
        println("Results exported to $(out_vtk).vtu with exact and error fields.")

        push!(errors_u, l2_u)
        push!(errors_p, l2_p)
        push!(errors_c, l2_c)

        println("  -> L2 Error U: $l2_u")
        println("  -> L2 Error P: $l2_p")
        println("  -> L2 Error C: $l2_c")
    end

    println("\n=== MMS Convergence Summary ===")
    for i in 1:length(Ns)
        println("  N=$(Ns[i]) (h=$(round(hs[i], digits=3))) | err_u = $(errors_u[i]) | err_p = $(errors_p[i]) | err_c = $(errors_c[i])")
    end

    open(joinpath(results_dir, "mms_convergence_data.json"), "w") do f
        JSON.print(f, Dict("h_vals"=>hs, "errors_u"=>errors_u, "errors_p"=>errors_p, "errors_c"=>errors_c), 4)
    end
    println("MMS Convergence discrete L2 data saved.")

    # Simple slope calculation (log-log)
    function eval_slope(errs, hs)
        return log(errs[end] / errs[end-1]) / log(hs[end] / hs[end-1])
    end

    slope_u = eval_slope(errors_u, hs)
    slope_p = eval_slope(errors_p, hs)
    slope_c = eval_slope(errors_c, hs)

    println("\n=== Asymptotic Convergence Rates (O(h^p)) ===")
    println("  Rate U (Velocity):      $slope_u")
    println("  Rate P (Pressure):      $slope_p")
    println("  Rate C (Concentration): $slope_c")
end

run_mms_convergence()
