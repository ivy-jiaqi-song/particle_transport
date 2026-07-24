haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

using HDF5
using CUDA
using PyPlot
using Random
using Statistics

include(joinpath(@__DIR__, "time_units.jl"))
include(joinpath(@__DIR__, "gpu_memory_planner.jl"))

const PIPELINE_ROOT = dirname(@__DIR__)
const C_LIGHT = 2.99792458e8
const M_P = 1.67262192369e-27
const GEV_TO_J = 1.0e9 * 1.602176634e-19

const DPP_FULL_CFG = Dict{Symbol, Any}(
    :trajectory_h5 => joinpath(PIPELINE_ROOT, "outputs", "campaigns", "0_5", "trajectory_cache", "phase_space_10000000_GeV.h5"),
    :cache_h5 => nothing,
    :cache_mode => :phase_space,
    :compute_backend => :gpu,
    :compute_precision => Float32,
    :accumulator_precision => Float32,
    :gpu_threads => 256,
    :gpu_lag_batch_size => 4,
    :gpu_memory_fraction => 0.75,
    :gpu_pipeline_buffers => 2,
    :particle_chunk_size => 128,
    :first_particle => 1,
    :n_particles_to_use => 10000,
    :particle_selection => :block_random,
    :particle_seed => 20260423,
    :particle_block_size => 128,
    :lag_mode => :uniform_samples,
    :lag_range_policy => :fixed,
    :lag_common_scope => :campaign,
    :lag_boundary_policy => :strict,
    :max_lag_boundary_relative_error => 0.0,
    :lag_min_gyroperiods => nothing,
    :lag_max_gyroperiods => nothing,
    :lag_stride_gyroperiods => nothing,
    :min_lag_steps => 1,
    :n_lag_samples => 40,
    :max_lag_steps => nothing,
    :lag_step_stride => 1,
    :n_energy_snapshots => 5,
    :n_energy_bins => 64,
    :save_raw_energy_snapshots => false,
    :output_dir => joinpath(PIPELINE_ROOT, "outputs", "dpp_full"),
    :output_h5 => nothing,
    :output_dpp_png => nothing,
    :use_usetex => false,
)

function parse_maybe_int(value::AbstractString)
    lowercase(strip(value)) in ("none", "nothing") && return nothing
    return parse(Int, value)
end

function parse_backend(value::AbstractString)
    backend = Symbol(lowercase(strip(value)))
    backend in (:auto, :gpu, :cpu) || error("compute_backend must be auto, gpu, or cpu.")
    return backend
end

function parse_precision(value::AbstractString)
    precision = lowercase(strip(value))
    precision == "float32" && return Float32
    precision == "float64" && return Float64
    error("compute_precision must be Float32 or Float64.")
end

function resolve_backend(cfg)
    requested = get(cfg, :compute_backend, :cpu)
    requested == :cpu && return :cpu
    if requested == :gpu
        CUDA.functional() || error("[dpp].compute_backend = gpu but CUDA.functional() is false on this machine.")
        return :gpu
    elseif requested == :auto
        return CUDA.functional() ? :gpu : :cpu
    end
    error("Unknown compute backend: " * string(requested) * ". Use gpu, cpu, or auto.")
end

function parse_particle_count(value::AbstractString)
    lowercase(strip(value)) == "all" && return nothing
    return parse(Int, value)
end

function parse_lag_mode(value::AbstractString)
    mode = Symbol(replace(lowercase(strip(value)), "-" => "_"))
    mode == :uniform && return :uniform_samples
    mode in (:uniform_samples, :stride) && return mode
    error("lag_mode must be uniform-samples or stride.")
end

function parse_particle_selection(value::AbstractString)
    selection = Symbol(replace(lowercase(strip(value)), "-" => "_"))
    selection in (:range, :random, :block_random, :random_blocks) || error("particle_selection must be range, random, or block-random.")
    selection == :random_blocks && return :block_random
    return selection
end

function parse_bool(value)
    normalized = lowercase(strip(String(value)))
    normalized in ("true", "yes", "1") && return true
    normalized in ("false", "no", "0") && return false
    error("Boolean values must be true or false.")
end

function parse_cli_config(args)
    cfg = Dict{Symbol, Any}(DPP_FULL_CFG)
    for argument in args
        if argument == "--help" || argument == "-h"
            println("Usage: julia src/compute_dpp_full.jl --trajectory-h5=PATH --output-dir=DIR [--lag-min-gyroperiods=A --lag-max-gyroperiods=B --n-lag-samples=N --lag-boundary-policy=strict|nearest --lag-range-policy=fixed|first-cache-step|common-cache-intersection --lag-common-scope=campaign|reference-group | legacy step controls]")
            exit(0)
        elseif startswith(argument, "--trajectory-h5=")
            cfg[:trajectory_h5] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--cache-h5=")
            cfg[:cache_h5] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--output-dir=")
            cfg[:output_dir] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--output-h5=")
            cfg[:output_h5] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--output-dpp-png=")
            cfg[:output_dpp_png] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--compute-backend=")
            cfg[:compute_backend] = parse_backend(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--compute-precision=")
            cfg[:compute_precision] = parse_precision(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--gpu-threads=")
            cfg[:gpu_threads] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--gpu-lag-batch-size=")
            cfg[:gpu_lag_batch_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--gpu-memory-fraction=")
            cfg[:gpu_memory_fraction] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-chunk-size=")
            cfg[:particle_chunk_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--first-particle=")
            cfg[:first_particle] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-particles=")
            cfg[:n_particles_to_use] = parse_particle_count(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-selection=")
            cfg[:particle_selection] = parse_particle_selection(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-seed=")
            cfg[:particle_seed] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-block-size=")
            cfg[:particle_block_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-mode=")
            cfg[:lag_mode] = parse_lag_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-range-policy=") || startswith(argument, "--dpp-lag-range-policy=")
            cfg[:lag_range_policy] = normalize_lag_range_policy(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-common-scope=") || startswith(argument, "--dpp-lag-common-scope=")
            cfg[:lag_common_scope] = normalize_lag_common_scope(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-boundary-policy=") || startswith(argument, "--dpp-lag-boundary-policy=")
            cfg[:lag_boundary_policy] = normalize_lag_boundary_policy(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--max-lag-boundary-relative-error=") || startswith(argument, "--dpp-max-lag-boundary-relative-error=")
            cfg[:max_lag_boundary_relative_error] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--min-lag-steps=")
            cfg[:min_lag_steps] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-min-gyroperiods=") || startswith(argument, "--dpp-lag-min-gyroperiods=")
            cfg[:lag_min_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-lag-samples=")
            cfg[:n_lag_samples] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--max-lag-steps=")
            cfg[:max_lag_steps] = parse_maybe_int(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-max-gyroperiods=") || startswith(argument, "--dpp-lag-max-gyroperiods=")
            cfg[:lag_max_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-step-stride=")
            cfg[:lag_step_stride] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-stride-gyroperiods=") || startswith(argument, "--dpp-lag-stride-gyroperiods=")
            cfg[:lag_stride_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-energy-snapshots=")
            cfg[:n_energy_snapshots] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-energy-bins=")
            cfg[:n_energy_bins] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--save-raw-energy-snapshots=")
            cfg[:save_raw_energy_snapshots] = parse_bool(split(argument, "=", limit=2)[2])
        elseif argument == "--save-raw-energy-snapshots"
            cfg[:save_raw_energy_snapshots] = true
        elseif startswith(argument, "--use-usetex=")
            cfg[:use_usetex] = parse_bool(split(argument, "=", limit=2)[2])
        else
            error("Unknown option: $argument. Use --help for supported options.")
        end
    end
    cfg[:output_h5] === nothing && (cfg[:output_h5] = joinpath(cfg[:output_dir], "dpp_full.h5"))
    cfg[:output_dpp_png] === nothing && (cfg[:output_dpp_png] = joinpath(cfg[:output_dir], "dpp_tau_curve_full.png"))
    return cfg
end

function build_selected_lag_steps(nsteps::Integer, cfg)
    min_lag = max(1, Int(get(cfg, :min_lag_steps, 1)))
    max_lag = cfg[:max_lag_steps] === nothing ? nsteps - 1 : min(Int(cfg[:max_lag_steps]), nsteps - 1)
    max_lag >= min_lag || error("No lag steps selected. Check min_lag_steps, max_lag_steps, and saved trajectory length.")
    if cfg[:lag_mode] == :uniform_samples
        n_lags = min(Int(cfg[:n_lag_samples]), max_lag - min_lag + 1)
        return unique(round.(Int, range(min_lag, max_lag, length=n_lags)))
    elseif cfg[:lag_mode] == :stride
        stride = Int(cfg[:lag_step_stride])
        stride >= 1 || error("lag_step_stride must be >= 1")
        return collect(min_lag:stride:max_lag)
    end
    error("Unknown lag_mode: $(cfg[:lag_mode])")
end

function build_particle_indices(total_particles::Integer, cfg)
    first_particle = Int(cfg[:first_particle])
    first_particle >= 1 || error("first_particle must be >= 1.")
    first_particle <= total_particles || error("first_particle exceeds total particle count.")
    available_count = total_particles - first_particle + 1
    requested_count = cfg[:n_particles_to_use] === nothing ? available_count : Int(cfg[:n_particles_to_use])
    requested_count > 0 || error("n_particles_to_use must be positive or all.")
    selected_count = min(available_count, requested_count)
    if cfg[:particle_selection] == :range
        return collect(first_particle:(first_particle + selected_count - 1))
    elseif cfg[:particle_selection] == :random
        rng = MersenneTwister(Int(cfg[:particle_seed]))
        offsets = sort!(randperm(rng, available_count)[1:selected_count])
        return first_particle .+ offsets .- 1
    elseif cfg[:particle_selection] == :block_random
        block_size = Int(cfg[:particle_block_size])
        block_size > 0 || error("particle_block_size must be positive.")
        block_starts = collect(first_particle:block_size:total_particles)
        block_count = min(length(block_starts), cld(selected_count, block_size))
        rng = MersenneTwister(Int(cfg[:particle_seed]))
        selected_blocks = sort!(randperm(rng, length(block_starts))[1:block_count])
        particle_indices = Int[]
        sizehint!(particle_indices, selected_count)
        for block_index in selected_blocks
            block_start = block_starts[block_index]
            block_stop = min(total_particles, block_start + block_size - 1)
            for particle_index in block_start:block_stop
                length(particle_indices) >= selected_count && break
                push!(particle_indices, particle_index)
            end
            length(particle_indices) >= selected_count && break
        end
        return particle_indices
    end
    error("Unknown particle_selection: $(cfg[:particle_selection])")
end

function is_contiguous(indices)
    length(indices) <= 1 && return true
    @inbounds for index in 2:length(indices)
        indices[index] == indices[index - 1] + 1 || return false
    end
    return true
end

function read_particle_batch(dataset, particle_indices)
    isempty(particle_indices) && error("Cannot read an empty particle batch.")
    if is_contiguous(particle_indices)
        return dataset[first(particle_indices):last(particle_indices), :, :]
    end
    output = Array{eltype(dataset), 3}(undef, length(particle_indices), size(dataset, 2), size(dataset, 3))
    output_offset = 1
    run_start = 1
    @inbounds while run_start <= length(particle_indices)
        run_stop = run_start
        while run_stop < length(particle_indices) && particle_indices[run_stop + 1] == particle_indices[run_stop] + 1
            run_stop += 1
        end
        source_first = particle_indices[run_start]
        source_last = particle_indices[run_stop]
        run_length = run_stop - run_start + 1
        output[output_offset:(output_offset + run_length - 1), :, :] = dataset[source_first:source_last, :, :]
        output_offset += run_length
        run_start = run_stop + 1
    end
    return output
end

function validate_phase_space_layout(momenta_ds, t_s, t_norm)
    size(momenta_ds, 2) == 3 || error("momenta dataset must have shape (particles, 3, steps).")
    nsteps = size(momenta_ds, 3)
    length(t_s) == nsteps || error("t_s length does not match saved momentum steps.")
    length(t_norm) == nsteps || error("t_norm length does not match saved momentum steps.")
    nsteps > 1 || error("Need at least two saved trajectory steps.")
    return nsteps
end

@inline function momentum_magnitude(px::Real, py::Real, pz::Real)
    return sqrt(Float64(px) * Float64(px) + Float64(py) * Float64(py) + Float64(pz) * Float64(pz))
end

@inline function kinetic_energy_gev_from_pmag(pmag::Real)
    isfinite(pmag) || return NaN
    gamma = sqrt(1.0 + (Float64(pmag) / (M_P * C_LIGHT))^2)
    return (gamma - 1.0) * M_P * C_LIGHT^2 / GEV_TO_J
end

function selected_energy_snapshot_indices(nsteps::Integer, count::Integer)
    snapshot_count = min(max(1, Int(count)), nsteps)
    return unique(round.(Int, range(1, nsteps, length=snapshot_count)))
end

function process_momenta_chunk_dpp_cpu!(momenta, lag_steps::Vector{Int}, p0::Float64, counts, sum_delta_p, sum_delta_p2, sum_delta_p_norm, sum_delta_p_norm2)
    n_particles = size(momenta, 1)
    nsteps = size(momenta, 3)
    thread_count = Threads.maxthreadid()
    local_counts = zeros(Int64, thread_count)
    local_sum_delta_p = zeros(Float64, thread_count)
    local_sum_delta_p2 = zeros(Float64, thread_count)
    local_sum_delta_p_norm = zeros(Float64, thread_count)
    local_sum_delta_p_norm2 = zeros(Float64, thread_count)
    @inbounds for (lag_index, lag_step) in enumerate(lag_steps)
        lag_step >= nsteps && continue
        last_start = nsteps - lag_step
        fill!(local_counts, 0)
        fill!(local_sum_delta_p, 0.0)
        fill!(local_sum_delta_p2, 0.0)
        fill!(local_sum_delta_p_norm, 0.0)
        fill!(local_sum_delta_p_norm2, 0.0)
        Threads.@threads for particle_index in 1:n_particles
            thread_index = Threads.threadid()
            for step_index in 1:last_start
                p_start = momentum_magnitude(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
                p_end = momentum_magnitude(momenta[particle_index, 1, step_index + lag_step], momenta[particle_index, 2, step_index + lag_step], momenta[particle_index, 3, step_index + lag_step])
                if !(isfinite(p_start) && isfinite(p_end))
                    continue
                end
                delta_p = p_end - p_start
                delta_p_norm = delta_p / p0
                local_counts[thread_index] += 1
                local_sum_delta_p[thread_index] += delta_p
                local_sum_delta_p2[thread_index] += delta_p * delta_p
                local_sum_delta_p_norm[thread_index] += delta_p_norm
                local_sum_delta_p_norm2[thread_index] += delta_p_norm * delta_p_norm
            end
        end
        counts[lag_index] += sum(local_counts)
        sum_delta_p[lag_index] += sum(local_sum_delta_p)
        sum_delta_p2[lag_index] += sum(local_sum_delta_p2)
        sum_delta_p_norm[lag_index] += sum(local_sum_delta_p_norm)
        sum_delta_p_norm2[lag_index] += sum(local_sum_delta_p_norm2)
    end
    return nothing
end

function fill_energy_snapshots!(energy_snapshots, momenta, snapshot_indices, particle_offset::Integer)
    n_particles = size(momenta, 1)
    @inbounds for (snapshot_position, step_index) in enumerate(snapshot_indices)
        for particle_index in 1:n_particles
            pmag = momentum_magnitude(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
            energy_snapshots[snapshot_position, particle_offset + particle_index - 1] = kinetic_energy_gev_from_pmag(pmag)
        end
    end
    return nothing
end

@inline function momentum_magnitude_gpu(px, py, pz)
    return sqrt(px * px + py * py + pz * pz)
end

@inline function kinetic_energy_gev_from_pmag_gpu(pmag)
    T = typeof(pmag)
    if !isfinite(pmag)
        return T(NaN)
    end
    gamma = sqrt(one(T) + (pmag / (T(M_P) * T(C_LIGHT)))^2)
    return (gamma - one(T)) * T(M_P) * T(C_LIGHT)^2 / T(GEV_TO_J)
end

function dpp_particle_partials_kernel!(partial_counts, partial_sum_delta_p, partial_sum_delta_p2, partial_sum_delta_p_norm, partial_sum_delta_p_norm2, momenta, lag_steps, p0)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    n_particles = size(momenta, 1)
    nsteps = size(momenta, 3)
    n_lags = length(lag_steps)
    total = n_particles * n_lags
    T = eltype(momenta)
    p0_t = T(p0)

    while index <= total
        lag_index = ((index - 1) % n_lags) + 1
        particle_index = ((index - 1) ÷ n_lags) + 1
        lag_step = Int(lag_steps[lag_index])
        if lag_step < nsteps
            last_start = nsteps - lag_step
            local_count = Int64(0)
            local_sum_delta_p = zero(T)
            local_sum_delta_p2 = zero(T)
            local_sum_delta_p_norm = zero(T)
            local_sum_delta_p_norm2 = zero(T)
            @inbounds for step_index in 1:last_start
                p_start = momentum_magnitude_gpu(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
                p_end = momentum_magnitude_gpu(momenta[particle_index, 1, step_index + lag_step], momenta[particle_index, 2, step_index + lag_step], momenta[particle_index, 3, step_index + lag_step])
                if isfinite(p_start) && isfinite(p_end)
                    delta_p = p_end - p_start
                    delta_p_norm = delta_p / p0_t
                    local_count += Int64(1)
                    local_sum_delta_p += delta_p
                    local_sum_delta_p2 += delta_p * delta_p
                    local_sum_delta_p_norm += delta_p_norm
                    local_sum_delta_p_norm2 += delta_p_norm * delta_p_norm
                end
            end
            if local_count > 0
                partial_counts[lag_index, particle_index] = local_count
                partial_sum_delta_p[lag_index, particle_index] = local_sum_delta_p
                partial_sum_delta_p2[lag_index, particle_index] = local_sum_delta_p2
                partial_sum_delta_p_norm[lag_index, particle_index] = local_sum_delta_p_norm
                partial_sum_delta_p_norm2[lag_index, particle_index] = local_sum_delta_p_norm2
            end
        end
        index += stride
    end
    return
end

function dpp_reduce_partials_kernel!(campaign_counts, campaign_sum_delta_p, campaign_sum_delta_p2, campaign_sum_delta_p_norm, campaign_sum_delta_p_norm2, partial_counts, partial_sum_delta_p, partial_sum_delta_p2, partial_sum_delta_p_norm, partial_sum_delta_p_norm2, lag_offset::Int32)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    n_lags = size(partial_counts, 1)
    n_particles = size(partial_counts, 2)
    total = n_lags * n_particles
    while index <= total
        local_lag = ((index - 1) % n_lags) + 1
        particle_index = ((index - 1) ÷ n_lags) + 1
        count = partial_counts[local_lag, particle_index]
        if count > Int64(0)
            lag_index = Int(lag_offset) + local_lag - 1
            CUDA.@atomic campaign_counts[lag_index] += count
            CUDA.@atomic campaign_sum_delta_p[lag_index] += partial_sum_delta_p[local_lag, particle_index]
            CUDA.@atomic campaign_sum_delta_p2[lag_index] += partial_sum_delta_p2[local_lag, particle_index]
            CUDA.@atomic campaign_sum_delta_p_norm[lag_index] += partial_sum_delta_p_norm[local_lag, particle_index]
            CUDA.@atomic campaign_sum_delta_p_norm2[lag_index] += partial_sum_delta_p_norm2[local_lag, particle_index]
        end
        index += stride
    end
    return
end

function energy_snapshots_kernel!(energy_snapshots, momenta, snapshot_indices)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = length(energy_snapshots)
    index > total && return
    n_snapshots = size(energy_snapshots, 1)
    snapshot_position = ((index - 1) % n_snapshots) + 1
    particle_index = ((index - 1) ÷ n_snapshots) + 1
    step_index = Int(snapshot_indices[snapshot_position])
    pmag = momentum_magnitude_gpu(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
    energy_snapshots[snapshot_position, particle_index] = Float64(kinetic_energy_gev_from_pmag_gpu(pmag))
    return
end

function process_momenta_chunk_dpp_gpu!(momenta, lag_steps::Vector{Int}, lag_range, p0::Float64, gpu_accumulators, cfg)
    T = get(cfg, :compute_precision, Float32)
    dmomenta = momenta isa CuArray ? momenta : CuArray(T.(momenta))
    lag_subset = lag_steps[lag_range]
    dlag_steps = CuArray(Int32.(lag_subset))
    n_lags = length(lag_subset)
    n_particles = size(momenta, 1)
    partial_counts = CUDA.zeros(Int64, n_lags, n_particles)
    partial_sum_delta_p = CUDA.zeros(T, n_lags, n_particles)
    partial_sum_delta_p2 = CUDA.zeros(T, n_lags, n_particles)
    partial_sum_delta_p_norm = CUDA.zeros(T, n_lags, n_particles)
    partial_sum_delta_p_norm2 = CUDA.zeros(T, n_lags, n_particles)
    threads = Int(get(cfg, :gpu_threads, 256))
    blocks = min(4096, cld(n_particles * n_lags, threads))
    @cuda threads=threads blocks=blocks dpp_particle_partials_kernel!(partial_counts, partial_sum_delta_p, partial_sum_delta_p2, partial_sum_delta_p_norm, partial_sum_delta_p_norm2, dmomenta, dlag_steps, p0)
    reduce_blocks = min(4096, cld(length(partial_counts), threads))
    @cuda threads=threads blocks=reduce_blocks dpp_reduce_partials_kernel!(
        gpu_accumulators.counts,
        gpu_accumulators.sum_delta_p,
        gpu_accumulators.sum_delta_p2,
        gpu_accumulators.sum_delta_p_norm,
        gpu_accumulators.sum_delta_p_norm2,
        partial_counts,
        partial_sum_delta_p,
        partial_sum_delta_p2,
        partial_sum_delta_p_norm,
        partial_sum_delta_p_norm2,
        Int32(first(lag_range)),
    )
    return dmomenta
end

function create_dpp_gpu_accumulators(n_lags::Integer)
    return (
        counts = CUDA.zeros(Int64, n_lags),
        sum_delta_p = CUDA.zeros(Float32, n_lags),
        sum_delta_p2 = CUDA.zeros(Float32, n_lags),
        sum_delta_p_norm = CUDA.zeros(Float32, n_lags),
        sum_delta_p_norm2 = CUDA.zeros(Float32, n_lags),
    )
end

function copy_dpp_gpu_accumulators!(gpu_accumulators, counts, sum_delta_p, sum_delta_p2, sum_delta_p_norm, sum_delta_p_norm2)
    counts .= Array(gpu_accumulators.counts)
    sum_delta_p .= Float64.(Array(gpu_accumulators.sum_delta_p))
    sum_delta_p2 .= Float64.(Array(gpu_accumulators.sum_delta_p2))
    sum_delta_p_norm .= Float64.(Array(gpu_accumulators.sum_delta_p_norm))
    sum_delta_p_norm2 .= Float64.(Array(gpu_accumulators.sum_delta_p_norm2))
    return nothing
end

function fill_energy_snapshots_gpu!(energy_snapshots, dmomenta, snapshot_indices, particle_offset::Integer, cfg)
    dindices = CuArray(Int32.(snapshot_indices))
    denergy = CUDA.fill(Float64(NaN), length(snapshot_indices), size(dmomenta, 1))
    threads = Int(get(cfg, :gpu_threads, 256))
    blocks = cld(length(denergy), threads)
    @cuda threads=threads blocks=blocks energy_snapshots_kernel!(denergy, dmomenta, dindices)
    CUDA.synchronize()
    energy_snapshots[:, particle_offset:(particle_offset + size(dmomenta, 1) - 1)] = Array(denergy)
    return nothing
end

function energy_histogram_kernel!(hist_counts, moment_sum, moment_sum2, moment_counts, momenta, snapshot_indices, emin, inv_width, n_bins::Int32)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    n_snapshots = length(snapshot_indices)
    n_particles = size(momenta, 1)
    total = n_snapshots * n_particles
    T = eltype(momenta)
    while index <= total
        snapshot_position = ((index - 1) % n_snapshots) + 1
        particle_index = ((index - 1) ÷ n_snapshots) + 1
        step_index = Int(snapshot_indices[snapshot_position])
        pmag = momentum_magnitude_gpu(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
        energy = kinetic_energy_gev_from_pmag_gpu(pmag)
        if isfinite(energy)
            raw = floor(Int32, (energy - T(emin)) * T(inv_width)) + Int32(1)
            if raw == n_bins + Int32(1)
                raw = n_bins
            end
            if Int32(1) <= raw <= n_bins
                CUDA.@atomic hist_counts[snapshot_position, raw] += Int64(1)
                CUDA.@atomic moment_sum[snapshot_position] += Float32(energy)
                CUDA.@atomic moment_sum2[snapshot_position] += Float32(energy * energy)
                CUDA.@atomic moment_counts[snapshot_position] += Int64(1)
            end
        end
        index += stride
    end
    return
end

function energy_range_cpu(momenta, snapshot_indices)
    emin = Inf
    emax = -Inf
    @inbounds for step_index in snapshot_indices, particle_index in 1:size(momenta, 1)
        pmag = momentum_magnitude(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
        energy = kinetic_energy_gev_from_pmag(pmag)
        if isfinite(energy)
            emin = min(emin, energy)
            emax = max(emax, energy)
        end
    end
    return emin, emax
end

function fill_energy_histograms_cpu!(hist_counts, moment_sum, moment_sum2, moment_counts, momenta, snapshot_indices, energy_edges)
    n_bins = length(energy_edges) - 1
    @inbounds for (snapshot_position, step_index) in enumerate(snapshot_indices)
        for particle_index in 1:size(momenta, 1)
            pmag = momentum_magnitude(momenta[particle_index, 1, step_index], momenta[particle_index, 2, step_index], momenta[particle_index, 3, step_index])
            energy = kinetic_energy_gev_from_pmag(pmag)
            isfinite(energy) || continue
            bin_index = searchsortedlast(energy_edges, energy)
            bin_index == length(energy_edges) && (bin_index = n_bins)
            if 1 <= bin_index <= n_bins
                hist_counts[snapshot_position, bin_index] += 1
                moment_sum[snapshot_position] += energy
                moment_sum2[snapshot_position] += energy * energy
                moment_counts[snapshot_position] += 1
            end
        end
    end
    return nothing
end

function fill_energy_histograms_gpu!(hist_accumulators, dmomenta, snapshot_indices, energy_edges, cfg)
    n_bins = length(energy_edges) - 1
    dindices = CuArray(Int32.(snapshot_indices))
    threads = Int(get(cfg, :gpu_threads, 256))
    blocks = min(4096, cld(length(snapshot_indices) * size(dmomenta, 1), threads))
    inv_width = n_bins / (energy_edges[end] - energy_edges[1])
    @cuda threads=threads blocks=blocks energy_histogram_kernel!(hist_accumulators.counts, hist_accumulators.sum, hist_accumulators.sum2, hist_accumulators.moment_counts, dmomenta, dindices, Float64(energy_edges[1]), Float64(inv_width), Int32(n_bins))
    return nothing
end

function create_energy_histogram_accumulators(n_snapshots::Integer, n_bins::Integer)
    return (
        counts = CUDA.zeros(Int64, n_snapshots, n_bins),
        sum = CUDA.zeros(Float32, n_snapshots),
        sum2 = CUDA.zeros(Float32, n_snapshots),
        moment_counts = CUDA.zeros(Int64, n_snapshots),
    )
end

function compute_dpp_arrays(counts, sum_delta_p, sum_delta_p2, sum_delta_p_norm, sum_delta_p_norm2, tau_s, tau_norm)
    n_lags = length(counts)
    mean_delta_p = fill(NaN, n_lags)
    mean_delta_p2 = fill(NaN, n_lags)
    mean_delta_p_norm = fill(NaN, n_lags)
    mean_delta_p_norm2 = fill(NaN, n_lags)
    dpp_raw_per_s = fill(NaN, n_lags)
    dpp_centered_per_s = fill(NaN, n_lags)
    dpp_raw_norm = fill(NaN, n_lags)
    dpp_centered_norm = fill(NaN, n_lags)
    @inbounds for lag_index in 1:n_lags
        count = counts[lag_index]
        count > 0 || continue
        delta_mean = sum_delta_p[lag_index] / count
        delta2_mean = sum_delta_p2[lag_index] / count
        delta_norm_mean = sum_delta_p_norm[lag_index] / count
        delta_norm2_mean = sum_delta_p_norm2[lag_index] / count
        centered_delta2 = max(0.0, delta2_mean - delta_mean * delta_mean)
        centered_delta_norm2 = max(0.0, delta_norm2_mean - delta_norm_mean * delta_norm_mean)
        mean_delta_p[lag_index] = delta_mean
        mean_delta_p2[lag_index] = delta2_mean
        mean_delta_p_norm[lag_index] = delta_norm_mean
        mean_delta_p_norm2[lag_index] = delta_norm2_mean
        dpp_raw_per_s[lag_index] = delta2_mean / (2.0 * Float64(tau_s[lag_index]))
        dpp_centered_per_s[lag_index] = centered_delta2 / (2.0 * Float64(tau_s[lag_index]))
        dpp_raw_norm[lag_index] = delta_norm2_mean / (2.0 * Float64(tau_norm[lag_index]))
        dpp_centered_norm[lag_index] = centered_delta_norm2 / (2.0 * Float64(tau_norm[lag_index]))
    end
    return mean_delta_p, mean_delta_p2, mean_delta_p_norm, mean_delta_p_norm2, dpp_raw_per_s, dpp_centered_per_s, dpp_raw_norm, dpp_centered_norm
end

function copy_cache_metadata!(file, cache_h5)
    cache_h5 === nothing && return nothing
    h5open(cache_h5, "r") do cache_file
        for key in ("energy_GeV", "dt_s", "dt_tOmega0", "dt_gyroperiods", "Omega0", "B0_T", "Omega0_reference_s_inv", "B0_reference_T", "reference_gyroperiod_s", "time_reference_name", "time_reference_mode", "time_reference_definition", "time_reference_B0_definition", "time_reference_energy_definition", "time_reference_source_mode", "time_reference_source_path", "time_reference_source_identity", "time_reference_field_subset", "requested_trajectory_duration_gyroperiods", "actual_trajectory_duration_gyroperiods", "actual_integration_duration_gyroperiods", "analysis_cache_duration_gyroperiods", "actual_trajectory_duration_tOmega0", "actual_trajectory_duration_s", "requested_integration_steps_per_gyroperiod", "actual_integration_steps_per_gyroperiod", "timestep_limited_by", "requested_trajectory_save_interval_gyroperiods", "actual_trajectory_save_interval_gyroperiods", "integration_step_count", "analysis_sample_count", "exact_final_state_stored", "trajectory_mode", "trajectory_field_source_path", "trajectory_field_source_identity", "n_particles", "trajectory_time_stride", "boundary_mode", "cache_mode", "momentum_unit", "injection_mode", "injection_mu0", "injection_position_mode", "injection_position", "injection_position_unit")
            haskey(cache_file, key) && !haskey(file, key) && (file[key] = read(cache_file[key]))
        end
    end
    return nothing
end

function save_dpp_full_h5(path_h5::AbstractString, cfg, lag_steps, tau_s, tau_norm, first_particle, last_particle, particle_indices, dpp_results, energy_snapshot_results; lag_grid=nothing)
    mkpath(dirname(path_h5))
    h5open(path_h5, "w") do file
        file["trajectory_h5"] = string(cfg[:trajectory_h5])
        file["cache_h5"] = string(get(cfg, :cache_h5, cfg[:trajectory_h5]))
        file["cache_mode"] = "phase_space"
        resolved_backend = get(cfg, :resolved_compute_backend, get(cfg, :compute_backend, :cpu))
        file["compute_backend"] = string(resolved_backend)
        file["requested_compute_backend"] = string(get(cfg, :compute_backend, resolved_backend))
        file["resolved_compute_backend"] = string(resolved_backend)
        file["compute_precision"] = string(get(cfg, :compute_precision, Float32))
        file["backend_version"] = resolved_backend == :gpu ? "gpu_dpp_partial_reduce_v2" : "cpu_dpp_reference_v1"
        file["accumulator_precision"] = string(get(cfg, :accumulator_precision, Float32))
        file["gpu_lag_batch_size"] = Int(get(cfg, :resolved_gpu_lag_batch_size, get(cfg, :gpu_lag_batch_size, 1)))
        file["gpu_memory_fraction"] = Float64(get(cfg, :gpu_memory_fraction, 1.0))
        file["save_raw_energy_snapshots"] = Bool(get(cfg, :save_raw_energy_snapshots, false))
        file["postprocess_pipeline_enabled"] = Bool(get(cfg, :gpu_pipeline_buffers, 1) >= 2 && resolved_backend == :gpu)
        file["source_cache_uniform_time_axis"] = true
        file["source_cache_identity"] = string(get(cfg, :cache_h5, cfg[:trajectory_h5]))
        file["particle_chunk_size"] = Int(cfg[:particle_chunk_size])
        file["first_particle"] = Int(first_particle)
        file["last_particle"] = Int(last_particle)
        file["n_particles_used"] = length(particle_indices)
        file["particle_selection"] = string(cfg[:particle_selection])
        file["particle_seed"] = Int(cfg[:particle_seed])
        file["particle_block_size"] = Int(cfg[:particle_block_size])
        file["particle_indices"] = particle_indices
        file["lag_mode"] = string(cfg[:lag_mode])
        file["lag_range_policy"] = string(get(cfg, :lag_range_policy, :fixed))
        file["lag_common_scope"] = string(get(cfg, :lag_common_scope, :campaign))
        file["lag_boundary_policy"] = string(get(cfg, :lag_boundary_policy, :strict))
        file["max_lag_boundary_relative_error"] = Float64[get(cfg, :max_lag_boundary_relative_error, 0.0)]
        file["n_lag_samples"] = length(lag_steps)
        file["requested_n_lag_samples"] = Int(cfg[:n_lag_samples])
        haskey(cfg, :lag_min_gyroperiods) && cfg[:lag_min_gyroperiods] !== nothing && (file["lag_min_gyroperiods"] = Float64[cfg[:lag_min_gyroperiods]])
        haskey(cfg, :lag_max_gyroperiods) && cfg[:lag_max_gyroperiods] !== nothing && (file["lag_max_gyroperiods"] = Float64[cfg[:lag_max_gyroperiods]])
        haskey(cfg, :lag_stride_gyroperiods) && cfg[:lag_stride_gyroperiods] !== nothing && (file["lag_stride_gyroperiods"] = Float64[cfg[:lag_stride_gyroperiods]])
        file["min_lag_steps"] = Int(get(cfg, :min_lag_steps, 1))
        file["lag_step_stride"] = Int(cfg[:lag_step_stride])
        file["max_lag_steps"] = cfg[:max_lag_steps] === nothing ? -1 : Int(cfg[:max_lag_steps])
        if lag_grid !== nothing
            file["source_cache_analysis_sample_count"] = Int[round(Int, lag_grid.cache_lag_max_gyroperiods / lag_grid.cache_save_interval_gyroperiods) + 1]
            file["source_cache_save_interval_gyroperiods"] = Float64[lag_grid.cache_save_interval_gyroperiods]
        end
        file["dpp_note"] = "Global scalar momentum diffusion uses p = sqrt(px^2 + py^2 + pz^2) and Delta p = p(t + tau) - p(t). Normalized D_pp uses Delta p / p0 and tau * Omega0."
        copy_cache_metadata!(file, get(cfg, :cache_h5, cfg[:trajectory_h5]))
        dpp_group = create_group(file, "dpp")
        dpp_group["lag_step"] = lag_steps
        dpp_group["lag_steps"] = lag_steps
        if lag_grid !== nothing
            dpp_group["requested_tau_gyroperiods"] = lag_grid.requested_tau_gyroperiods
            dpp_group["common_requested_tau_gyroperiods"] = lag_grid.common_requested_tau_gyroperiods
            dpp_group["tau_gyroperiods"] = lag_grid.tau_gyroperiods
            dpp_group["lag_mapping_error_gyroperiods"] = lag_grid.lag_mapping_error_gyroperiods
            file["lag_mapping_max_error_gyroperiods"] = Float64[lag_grid.max_lag_mapping_error_gyroperiods]
            file["lag_mapping_max_relative_error"] = Float64[lag_grid.max_lag_mapping_relative_error]
            file["requested_lag_count"] = Int[lag_grid.requested_lag_count]
            file["unique_lag_count"] = Int[lag_grid.unique_lag_count]
            file["duplicate_lag_mapping_count"] = Int[lag_grid.duplicate_lag_mapping_count]
            file["duplicate_lag_mapping_fraction"] = Float64[lag_grid.duplicate_lag_mapping_fraction]
            file["cache_lag_min_gyroperiods"] = Float64[lag_grid.cache_lag_min_gyroperiods]
            file["cache_lag_max_gyroperiods"] = Float64[lag_grid.cache_lag_max_gyroperiods]
            file["cache_save_interval_gyroperiods"] = Float64[lag_grid.cache_save_interval_gyroperiods]
            file["configured_lag_min_gyroperiods"] = Float64[lag_grid.configured_lag_min_gyroperiods]
            file["configured_lag_max_gyroperiods"] = Float64[lag_grid.configured_lag_max_gyroperiods]
            file["common_cache_lag_min_gyroperiods"] = Float64[lag_grid.common_cache_lag_min_gyroperiods]
            file["common_cache_lag_max_gyroperiods"] = Float64[lag_grid.common_cache_lag_max_gyroperiods]
            file["effective_lag_min_gyroperiods"] = Float64[lag_grid.effective_lag_min_gyroperiods]
            file["effective_lag_max_gyroperiods"] = Float64[lag_grid.effective_lag_max_gyroperiods]
            file["lag_comparison_group_identity"] = lag_grid.lag_comparison_group_identity
            file["preflight_job_id"] = lag_grid.preflight_job_id
            file["preflight_reference_group_id"] = lag_grid.preflight_reference_group_id
            file["lag_group_member_count"] = Int[lag_grid.lag_group_member_count]
            file["lag_group_member_modes"] = lag_grid.lag_group_member_modes
            file["lag_group_member_energies_GeV"] = lag_grid.lag_group_member_energies_GeV
            file["lag_grid_source"] = string(lag_grid.lag_source)
        else
            dpp_group["requested_tau_gyroperiods"] = tOmega0_to_gyroperiods.(tau_norm)
            dpp_group["tau_gyroperiods"] = tOmega0_to_gyroperiods.(tau_norm)
            dpp_group["lag_mapping_error_gyroperiods"] = zeros(Float64, length(lag_steps))
        end
        dpp_group["tau_s"] = tau_s
        dpp_group["tau_norm"] = tau_norm
        dpp_group["p0_kg_m_per_s"] = Float64[dpp_results.p0]
        dpp_group["count_pairs_full"] = dpp_results.counts
        dpp_group["sum_delta_p"] = dpp_results.sum_delta_p
        dpp_group["sum_delta_p2"] = dpp_results.sum_delta_p2
        dpp_group["sum_delta_p_over_p0"] = dpp_results.sum_delta_p_norm
        dpp_group["sum_delta_p_over_p0_2"] = dpp_results.sum_delta_p_norm2
        dpp_group["mean_delta_p"] = dpp_results.mean_delta_p
        dpp_group["mean_delta_p2"] = dpp_results.mean_delta_p2
        dpp_group["mean_delta_p_over_p0"] = dpp_results.mean_delta_p_norm
        dpp_group["mean_delta_p_over_p0_2"] = dpp_results.mean_delta_p_norm2
        dpp_group["D_pp_raw_per_s"] = dpp_results.dpp_raw_per_s
        dpp_group["D_pp_centered_per_s"] = dpp_results.dpp_centered_per_s
        dpp_group["D_pp_raw_norm"] = dpp_results.dpp_raw_norm
        dpp_group["D_pp_centered_norm"] = dpp_results.dpp_centered_norm
        energy_group = create_group(file, "energy_snapshots")
        energy_group["snapshot_step_index"] = energy_snapshot_results.indices
        energy_group["snapshot_t_s"] = energy_snapshot_results.t_s
        energy_group["snapshot_t_norm"] = energy_snapshot_results.t_norm
        energy_group["snapshot_t_gyroperiods"] = energy_snapshot_results.t_gyroperiods
        energy_group["energy_bin_edges_GeV"] = energy_snapshot_results.energy_bin_edges
        energy_group["energy_histogram_counts"] = energy_snapshot_results.histogram_counts
        energy_group["energy_count"] = energy_snapshot_results.counts
        energy_group["energy_mean_GeV"] = energy_snapshot_results.mean
        energy_group["energy_std_GeV"] = energy_snapshot_results.std
        if energy_snapshot_results.energy_gev !== nothing
            energy_group["energy_GeV"] = energy_snapshot_results.energy_gev
        end
        energy_group["particle_indices"] = particle_indices
    end
    return nothing
end

function plot_dpp_tau_curve(path_png::AbstractString, tau_gyroperiods, dpp_raw_norm, dpp_centered_norm; use_usetex::Bool=false)
    mkpath(dirname(path_png))
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(7, 4))
    plot(tau_gyroperiods, dpp_centered_norm, "o-", color="black", linewidth=1.5, markersize=4, label="centered")
    plot(tau_gyroperiods, dpp_raw_norm, "s--", color="tab:blue", linewidth=1.2, markersize=3, label="raw")
    xlabel("Lag [reference gyroperiods]")
    ylabel(raw"$D_{pp}/(p_0^2\Omega_0)$")
    title(raw"Global $D_{pp}(\tau)$")
    grid(true, alpha=0.3)
    legend(frameon=false, fontsize=8)
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
    return nothing
end

function run_dpp_full(cfg)
    mkpath(cfg[:output_dir])
    mkpath(dirname(cfg[:output_h5]))
    mkpath(dirname(cfg[:output_dpp_png]))
    backend = resolve_backend(cfg)
    cfg[:resolved_compute_backend] = backend
    h5open(cfg[:trajectory_h5], "r") do trajectory_file
        haskey(trajectory_file, "momenta") || error("D_pp requires a phase-space cache with a momenta dataset: " * string(cfg[:trajectory_h5]))
        momenta_dataset = trajectory_file["momenta"]
        t_s = Float64.(read(trajectory_file["t_s"]))
        t_norm = Float64.(read(trajectory_file["t_norm"]))
        t_gyroperiods = haskey(trajectory_file, "t_gyroperiods") ? Float64.(read(trajectory_file["t_gyroperiods"])) : t_gyroperiods_from_axes(t_s, t_norm)
        nsteps = validate_phase_space_layout(momenta_dataset, t_s, t_norm)
        validate_time_axes(t_s, t_norm, t_gyroperiods; key_name="D_pp trajectory time axes in " * string(cfg[:trajectory_h5]), require_uniform=true)
        total_particles = size(momenta_dataset, 1)
        particle_indices = build_particle_indices(total_particles, cfg)
        first_particle = first(particle_indices)
        last_particle = last(particle_indices)
        selected_particle_count = length(particle_indices)
        selected_particle_count > 0 || error("No particles selected.")
        lag_grid = resolve_lag_grid(cfg, t_gyroperiods)
        if lag_grid.duplicate_lag_mapping_count > 0 && lag_grid.duplicate_lag_mapping_count / lag_grid.requested_lag_count > 0.10
            @warn "Requested D_pp lag grid collapsed substantially after mapping to cached-step offsets." requested=lag_grid.requested_lag_count unique=lag_grid.unique_lag_count duplicates=lag_grid.duplicate_lag_mapping_count
        end
        lag_steps = lag_grid.lag_steps
        tau_s = [Float64(t_s[lag_step + 1] - t_s[1]) for lag_step in lag_steps]
        tau_norm = Float64.(lag_grid.tau_norm)
        tau_gyroperiods = Float64.(lag_grid.tau_gyroperiods)
        n_lags = length(lag_steps)
        p0 = momentum_magnitude(momenta_dataset[particle_indices[1], 1, 1], momenta_dataset[particle_indices[1], 2, 1], momenta_dataset[particle_indices[1], 3, 1])
        isfinite(p0) && p0 > 0.0 || error("Cannot compute D_pp normalization: first selected particle has invalid initial momentum.")
        dpp_counts = zeros(Int64, n_lags)
        dpp_sum_delta_p = zeros(Float64, n_lags)
        dpp_sum_delta_p2 = zeros(Float64, n_lags)
        dpp_sum_delta_p_norm = zeros(Float64, n_lags)
        dpp_sum_delta_p_norm2 = zeros(Float64, n_lags)
        snapshot_indices = selected_energy_snapshot_indices(nsteps, Int(get(cfg, :n_energy_snapshots, 5)))
        save_raw_energy = Bool(get(cfg, :save_raw_energy_snapshots, false))
        energy_snapshots = save_raw_energy ? fill(NaN, length(snapshot_indices), selected_particle_count) : nothing
        n_energy_bins = Int(get(cfg, :n_energy_bins, 64))
        n_energy_bins > 0 || error("n_energy_bins must be positive.")
        chunk_size = min(Int(cfg[:particle_chunk_size]), selected_particle_count)
        memory_plan = backend == :gpu ? resolve_gpu_work_plan(
            cfg;
            particle_chunk_size=chunk_size,
            lag_count=n_lags,
            n_bins=1,
            bytes_per_particle_lag=sizeof(Int64) + 4 * sizeof(Float32),
            base_bytes=sizeof(Float32) * chunk_size * 3 * nsteps,
        ) : nothing
        backend == :gpu && print_gpu_work_plan("D_pp", memory_plan)
        backend == :gpu && (cfg[:resolved_gpu_lag_batch_size] = memory_plan.lag_batch_size)
        chunk_size = backend == :gpu ? min(memory_plan.particle_chunk_size, selected_particle_count) : chunk_size
        chunk_size > 0 || error("particle_chunk_size must be positive.")
        nchunks = cld(selected_particle_count, chunk_size)
        lag_ranges = backend == :gpu ? gpu_lag_batches(lag_steps, memory_plan.lag_batch_size) : [1:n_lags]
        gpu_accumulators = backend == :gpu ? create_dpp_gpu_accumulators(n_lags) : nothing
        hist_gpu_accumulators = backend == :gpu ? create_energy_histogram_accumulators(length(snapshot_indices), n_energy_bins) : nothing

        energy_min = Inf
        energy_max = -Inf
        for chunk_id in 1:nchunks
            selection_first = (chunk_id - 1) * chunk_size + 1
            selection_last = min(selected_particle_count, selection_first + chunk_size - 1)
            chunk_indices = particle_indices[selection_first:selection_last]
            momenta = read_particle_batch(momenta_dataset, chunk_indices)
            local_min, local_max = energy_range_cpu(momenta, snapshot_indices)
            energy_min = min(energy_min, local_min)
            energy_max = max(energy_max, local_max)
        end
        isfinite(energy_min) && isfinite(energy_max) || error("Cannot build energy histogram: no finite snapshot energies.")
        energy_min == energy_max && (energy_max = energy_min + max(abs(energy_min), 1.0f0) * 1.0e-6)
        energy_edges = collect(range(Float64(energy_min), Float64(energy_max), length=n_energy_bins + 1))
        energy_hist_counts = zeros(Int64, length(snapshot_indices), n_energy_bins)
        energy_moment_counts = zeros(Int64, length(snapshot_indices))
        energy_sum = zeros(Float64, length(snapshot_indices))
        energy_sum2 = zeros(Float64, length(snapshot_indices))
        println("D_pp phase-space HDF5: ", cfg[:trajectory_h5])
        println("D_pp particle selection: ", cfg[:particle_selection])
        println("D_pp particles selected: ", selected_particle_count, ", index span ", first_particle, "-", last_particle)
        println("D_pp lag count: ", n_lags, " from ", first(lag_steps), " to ", last(lag_steps))
        println("D_pp particle chunk: ", chunk_size)
        println("D_pp chunks: ", nchunks)
        println("D_pp energy snapshots: ", length(snapshot_indices))
        println("D_pp accumulation backend: ", backend)
        for chunk_id in 1:nchunks
            selection_first = (chunk_id - 1) * chunk_size + 1
            selection_last = min(selected_particle_count, selection_first + chunk_size - 1)
            chunk_indices = particle_indices[selection_first:selection_last]
            println("D_pp chunk ", chunk_id, "/", nchunks, ": ", length(chunk_indices), " selected particles, index span ", first(chunk_indices), "-", last(chunk_indices))
            momenta = read_particle_batch(momenta_dataset, chunk_indices)
            if backend == :gpu
                dmomenta = CuArray(cfg[:compute_precision].(momenta))
                for lag_range in lag_ranges
                    process_momenta_chunk_dpp_gpu!(dmomenta, lag_steps, lag_range, p0, gpu_accumulators, cfg)
                end
                fill_energy_histograms_gpu!(hist_gpu_accumulators, dmomenta, snapshot_indices, energy_edges, cfg)
                save_raw_energy && fill_energy_snapshots_gpu!(energy_snapshots, dmomenta, snapshot_indices, selection_first, cfg)
            else
                process_momenta_chunk_dpp_cpu!(momenta, lag_steps, p0, dpp_counts, dpp_sum_delta_p, dpp_sum_delta_p2, dpp_sum_delta_p_norm, dpp_sum_delta_p_norm2)
                fill_energy_histograms_cpu!(energy_hist_counts, energy_sum, energy_sum2, energy_moment_counts, momenta, snapshot_indices, energy_edges)
                save_raw_energy && fill_energy_snapshots!(energy_snapshots, momenta, snapshot_indices, selection_first)
            end
        end
        if backend == :gpu
            CUDA.synchronize()
            copy_dpp_gpu_accumulators!(gpu_accumulators, dpp_counts, dpp_sum_delta_p, dpp_sum_delta_p2, dpp_sum_delta_p_norm, dpp_sum_delta_p_norm2)
            energy_hist_counts .= Array(hist_gpu_accumulators.counts)
            energy_sum .= Float64.(Array(hist_gpu_accumulators.sum))
            energy_sum2 .= Float64.(Array(hist_gpu_accumulators.sum2))
            energy_moment_counts .= Array(hist_gpu_accumulators.moment_counts)
        end
        energy_mean = [energy_moment_counts[i] > 0 ? energy_sum[i] / energy_moment_counts[i] : NaN for i in eachindex(energy_moment_counts)]
        energy_std = [energy_moment_counts[i] > 1 ? sqrt(max(0.0, energy_sum2[i] / energy_moment_counts[i] - energy_mean[i]^2)) : NaN for i in eachindex(energy_moment_counts)]
        mean_delta_p, mean_delta_p2, mean_delta_p_norm, mean_delta_p_norm2, dpp_raw_per_s, dpp_centered_per_s, dpp_raw_norm, dpp_centered_norm = compute_dpp_arrays(dpp_counts, dpp_sum_delta_p, dpp_sum_delta_p2, dpp_sum_delta_p_norm, dpp_sum_delta_p_norm2, tau_s, tau_norm)
        dpp_results = (
            p0 = p0,
            counts = dpp_counts,
            sum_delta_p = dpp_sum_delta_p,
            sum_delta_p2 = dpp_sum_delta_p2,
            sum_delta_p_norm = dpp_sum_delta_p_norm,
            sum_delta_p_norm2 = dpp_sum_delta_p_norm2,
            mean_delta_p = mean_delta_p,
            mean_delta_p2 = mean_delta_p2,
            mean_delta_p_norm = mean_delta_p_norm,
            mean_delta_p_norm2 = mean_delta_p_norm2,
            dpp_raw_per_s = dpp_raw_per_s,
            dpp_centered_per_s = dpp_centered_per_s,
            dpp_raw_norm = dpp_raw_norm,
            dpp_centered_norm = dpp_centered_norm,
        )
        energy_snapshot_results = (
            indices = snapshot_indices,
            t_s = Float64.(t_s[snapshot_indices]),
            t_norm = Float64.(t_norm[snapshot_indices]),
            t_gyroperiods = Float64.(t_gyroperiods[snapshot_indices]),
            energy_bin_edges = energy_edges,
            histogram_counts = energy_hist_counts,
            counts = energy_moment_counts,
            mean = energy_mean,
            std = energy_std,
            energy_gev = energy_snapshots,
        )
        save_dpp_full_h5(cfg[:output_h5], cfg, lag_steps, tau_s, tau_norm, first_particle, last_particle, particle_indices, dpp_results, energy_snapshot_results; lag_grid=lag_grid)
        println("Saved D_pp HDF5 to ", cfg[:output_h5])
        plot_dpp_tau_curve(cfg[:output_dpp_png], tau_gyroperiods, dpp_raw_norm, dpp_centered_norm; use_usetex=cfg[:use_usetex])
        println("Saved D_pp tau curve to ", cfg[:output_dpp_png])
    end
    return nothing
end

function main()
    cfg = parse_cli_config(ARGS)
    run_dpp_full(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
