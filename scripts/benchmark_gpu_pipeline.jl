using Dates
using InteractiveUtils

const ROOT = dirname(@__DIR__)

function print_usage()
    println("""
    Usage:
      julia scripts/benchmark_gpu_pipeline.jl --config=PATH [--smoke]
      julia scripts/benchmark_gpu_pipeline.jl --synthetic [--particles=N --saved-times=N --lags=N --mu-bins=N --particle-chunk-size=N --lag-batch-size=N]

    The config mode runs the normal pipeline and reports wall time. Synthetic mode
    builds deterministic in-memory mu and momentum fixtures and reports stage
    timings without external simulation files.
    """)
end

function int_arg(prefix, default)
    value = argument_value(prefix)
    value === nothing && return default
    return parse(Int, value)
end

function synthetic_mu(particle_count, saved_time_count)
    mu = Matrix{Float32}(undef, saved_time_count, particle_count)
    @inbounds for particle in 1:particle_count, step in 1:saved_time_count
        mu[step, particle] = Float32(sin(0.017 * step + 0.003 * particle))
    end
    return mu
end

function synthetic_momenta(particle_count, saved_time_count)
    momenta = Array{Float32, 3}(undef, particle_count, 3, saved_time_count)
    @inbounds for particle in 1:particle_count, step in 1:saved_time_count
        base = Float32(1.0f-19 * (1.0f0 + 0.001f0 * particle))
        momenta[particle, 1, step] = base * (1.0f0 + 0.01f0 * sin(Float32(0.01 * step)))
        momenta[particle, 2, step] = base * (0.2f0 + 0.01f0 * cos(Float32(0.02 * step)))
        momenta[particle, 3, step] = base * (0.1f0 + 0.01f0 * sin(Float32(0.03 * step)))
    end
    return momenta
end

function benchmark_synthetic()
    particle_count = int_arg("--particles=", 2048)
    saved_time_count = int_arg("--saved-times=", 256)
    lag_count = int_arg("--lags=", 16)
    mu_bin_count = int_arg("--mu-bins=", 24)
    particle_chunk_size = int_arg("--particle-chunk-size=", 512)
    lag_batch_size = int_arg("--lag-batch-size=", 4)
    lag_steps = unique(round.(Int, range(1, saved_time_count - 1, length=lag_count)))
    mu_edges = collect(Float32, range(0.0f0, 1.0f0, length=mu_bin_count + 1))
    println("  synthetic particles      = ", particle_count)
    println("  synthetic saved times    = ", saved_time_count)
    println("  synthetic lag count      = ", length(lag_steps))
    println("  synthetic mu bins        = ", mu_bin_count)
    println("  particle chunk size      = ", particle_chunk_size)
    println("  lag batch size           = ", lag_batch_size)

    mu = nothing
    momenta = nothing
    t_fixture = @elapsed begin
        mu = synthetic_mu(particle_count, saved_time_count)
        momenta = synthetic_momenta(particle_count, saved_time_count)
    end

    dmumu_counts = zeros(Int64, mu_bin_count, length(lag_steps))
    dmumu_sum = zeros(Float64, mu_bin_count, length(lag_steps))
    dmumu_sum2 = zeros(Float64, mu_bin_count, length(lag_steps))
    t_dmumu = @elapsed begin
        @inbounds for (lag_index, lag) in enumerate(lag_steps)
            last_start = saved_time_count - lag
            for particle in 1:particle_count, step in 1:last_start
                a = mu[step, particle]
                b = mu[step + lag, particle]
                delta = b - a
                bin = searchsortedlast(mu_edges, abs(a))
                bin == length(mu_edges) && (bin = mu_bin_count)
                if 1 <= bin <= mu_bin_count
                    dmumu_counts[bin, lag_index] += 1
                    dmumu_sum[bin, lag_index] += delta
                    dmumu_sum2[bin, lag_index] += delta * delta
                end
            end
        end
    end

    dpp_counts = zeros(Int64, length(lag_steps))
    dpp_sum2 = zeros(Float64, length(lag_steps))
    t_dpp = @elapsed begin
        @inbounds for (lag_index, lag) in enumerate(lag_steps)
            last_start = saved_time_count - lag
            for particle in 1:particle_count, step in 1:last_start
                p0 = sqrt(sum(abs2, momenta[particle, :, step]))
                p1 = sqrt(sum(abs2, momenta[particle, :, step + lag]))
                delta = p1 - p0
                dpp_counts[lag_index] += 1
                dpp_sum2[lag_index] += delta * delta
            end
        end
    end

    pair_visits = sum(saved_time_count .- lag_steps) * particle_count
    println("  fixture build [s]        = ", t_fixture)
    println("  D_mumu CPU baseline [s]  = ", t_dmumu)
    println("  D_pp CPU baseline [s]    = ", t_dpp)
    println("  valid D_mumu pairs/s     = ", pair_visits / t_dmumu)
    println("  valid D_pp pairs/s       = ", pair_visits / t_dpp)
    println("  histogram count checksum = ", sum(dmumu_counts))
    println("  D_pp count checksum      = ", sum(dpp_counts))
    println("  total wall time [s]      = ", t_fixture + t_dmumu + t_dpp)
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
        benchmark_synthetic()
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
