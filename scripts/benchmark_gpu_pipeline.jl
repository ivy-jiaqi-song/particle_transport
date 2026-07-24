using Dates
using InteractiveUtils

const ROOT = dirname(@__DIR__)

function print_usage()
    println("""
    Usage:
      julia scripts/benchmark_gpu_pipeline.jl --config=PATH [--smoke]
      julia scripts/benchmark_gpu_pipeline.jl --synthetic [--smoke]

    The config mode runs the normal pipeline and reports wall time. Synthetic mode
    currently records environment metadata and exits with a clear message; it is a
    placeholder for generated fixture construction on machines with the full Julia
    dependency stack installed.
    """)
end

function argument_value(prefix)
    for argument in ARGS
        startswith(argument, prefix) && return split(argument, "=", limit=2)[2]
    end
    return nothing
end

function print_environment()
    println("GPU pipeline benchmark")
    println("  timestamp               = ", Dates.now())
    println("  Julia                   = ", VERSION)
    println("  threads                 = ", Threads.nthreads())
    try
        @eval using CUDA
        println("  CUDA functional         = ", CUDA.functional())
        if CUDA.functional()
            println("  GPU model               = ", CUDA.name(CUDA.device()))
            println("  CUDA runtime            = ", CUDA.runtime_version())
            println("  CUDA driver             = ", CUDA.driver_version())
            try
                @eval using Pkg
                deps = Pkg.dependencies()
                cuda_pkg = first((pkg for pkg in values(deps) if pkg.name == "CUDA"), nothing)
                cuda_pkg !== nothing && println("  CUDA.jl                 = ", cuda_pkg.version)
            catch err
                println("  CUDA.jl                 = unavailable (", err, ")")
            end
        end
    catch err
        println("  CUDA                   = unavailable (", err, ")")
    end
end

function main()
    any(arg -> arg in ("--help", "-h"), ARGS) && (print_usage(); return)
    print_environment()

    config_path = argument_value("--config=")
    synthetic = any(arg -> arg == "--synthetic", ARGS)
    smoke = any(arg -> arg == "--smoke", ARGS)

    if synthetic
        println("  synthetic fixture        = skipped; generated fixtures require the full CUDA/HDF5 dependency stack")
        println("  total wall time [s]      = NaN")
        return
    end
    config_path === nothing && error("Pass --config=PATH or --synthetic. Use --help for usage.")

    include(joinpath(ROOT, "src", "multimode_multienergy_cache_pipeline.jl"))
    empty!(ARGS)
    push!(ARGS, "--config=" * config_path)
    smoke && push!(ARGS, "--smoke")
    elapsed = @elapsed begin
        cfg = runtime_config()
        println("  compute precision       = ", cfg[:trajectory_overrides][:precision])
        println("  output precision        = ", cfg[:trajectory_overrides][:trajectory_output_precision])
        println("  D_mumu backend          = ", cfg[:dmumu_overrides][:compute_backend])
        println("  D_mumu lag batch        = ", get(cfg[:dmumu_overrides], :gpu_lag_batch_size, nothing))
        println("  D_pp backend            = ", get(cfg[:dpp_overrides], :compute_backend, :cpu))
        println("  D_pp lag batch          = ", get(cfg[:dpp_overrides], :gpu_lag_batch_size, nothing))
        run_mode_campaigns(cfg)
    end
    println("  total wall time [s]     = ", elapsed)
end

main()
