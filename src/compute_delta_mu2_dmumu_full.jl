haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

include(joinpath(@__DIR__, "compute_delta_mu2_curve.jl"))
include(joinpath(@__DIR__, "gpu_memory_planner.jl"))

using Random
using Printf

const COMBINED_FULL_CFG = Dict{Symbol, Any}(
    :trajectory_h5 => joinpath(PIPELINE_ROOT, "outputs", "campaigns", "0_5", "trajectory_cache","phase_space_10000000_GeV.h5"),
    :turbulence_h5 => raw"/data/multiphase/MP_WeakB_0_5tcs.h5",
    :B_paths => ("i_mag_field", "j_mag_field", "k_mag_field"),
    :box_length_pc => 200.0,
    :field_subset => nothing,
    :compute_backend => :gpu,
    :compute_precision => Float32,
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
    :dmumu_start_mode => :sliding,
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
    :n_mu_bins => 24,
    :mu_bin_abs => true,
    :mu_min => 0.0,
    :mu_max => 1.0,
    :min_count_per_cell => 20,
    :max_pair_visits_without_allow => 5.0e9,
    :allow_huge => true,
    :output_dir => joinpath(PIPELINE_ROOT, "outputs", "campaigns", "0_5", "10000000_GeV", "full_test"),
    :output_h5 => nothing,
    :output_delta_png => nothing,
    :output_heatmap_png => nothing,
    :output_collapsed_png => nothing,
    :use_usetex => false,
)

function print_usage()
    println("""
    Usage:
      julia src/compute_delta_mu2_dmumu_full.jl [options]

    This standalone postprocessor reconstructs mu(t) once per particle chunk and
    uses that same chunk to accumulate both:
      - global <(Delta mu)^2>(tau)
      - D_mumu(mu,tau) and tau-averaged D_mumu(mu)

    Important: sliding D_mumu uses every valid start time for each selected lag.
    Injection mode uses only mu(0) -> mu(tau) for each selected particle and lag.
    Neither mode randomly samples pairs.

    Common options:
      --trajectory-h5=PATH
      --turbulence-h5=PATH
      --output-dir=DIR
      --compute-backend=auto|gpu|cpu
      --n-particles=all|N
      --first-particle=N
      --particle-selection=range|random|block-random
      --particle-seed=N
      --particle-block-size=N
      --particle-chunk-size=N
      --dmumu-start-mode=sliding|injection
      --min-lag-steps=N
      --lag-min-gyroperiods=VALUE
      --n-lag-samples=N
      --max-lag-steps=N|none
      --lag-max-gyroperiods=VALUE
      --lag-mode=uniform|stride
      --lag-range-policy=fixed|first-cache-step|common-cache-intersection
      --lag-common-scope=campaign|reference-group
      --lag-boundary-policy=strict|nearest
      --max-lag-boundary-relative-error=VALUE
      --lag-step-stride=N
      --lag-stride-gyroperiods=VALUE
      --n-mu-bins=N
      --mu-bin-abs
      --no-mu-bin-abs
      --mu-min=VALUE
      --mu-max=VALUE
      --field-subset=none|NX,NY,NZ
      --allow-huge
      --smoke

    Safety:
      Large D_mumu runs are blocked unless --allow-huge is supplied.
    """)
end

function parse_maybe_int(value::AbstractString)
    lowercase(strip(value)) in ("none", "nothing") && return nothing
    return parse(Int, value)
end

function parse_particle_count(value::AbstractString)
    lowercase(strip(value)) == "all" && return nothing
    return parse(Int, value)
end

function parse_field_subset(value::AbstractString)
    cleaned_value = lowercase(strip(value))
    cleaned_value in ("none", "nothing") && return nothing
    parts = split(value, ",")
    length(parts) == 3 || error("--field-subset must be none or NX,NY,NZ")
    return Tuple(parse.(Int, parts))
end

function parse_backend(value::AbstractString)
    backend = Symbol(lowercase(strip(value)))
    backend in (:auto, :gpu, :cpu) || error("--compute-backend must be auto, gpu, or cpu")
    return backend
end

function parse_particle_selection(value::AbstractString)
    normalized_value = replace(lowercase(strip(value)), "-" => "_")
    selection = Symbol(normalized_value)
    selection in (:range, :random, :block_random, :random_blocks) || error("--particle-selection must be range, random, or block-random")
    selection == :random_blocks && return :block_random
    return selection
end

function parse_precision(value::AbstractString)
    precision = lowercase(strip(value))
    precision == "float32" && return Float32
    precision == "float64" && return Float64
    error("--compute-precision must be Float32 or Float64")
end

function parse_lag_mode(value::AbstractString)
    mode = Symbol(replace(lowercase(strip(value)), "-" => "_"))
    if mode == :uniform
        return :uniform_samples
    elseif mode in (:uniform_samples, :stride)
        return mode
    end
    error("--lag-mode must be uniform or stride")
end

function parse_bool(value)
    normalized_value = lowercase(strip(String(value)))
    normalized_value in ("true", "yes", "1") && return true
    normalized_value in ("false", "no", "0") && return false
    error("Boolean values must be true or false.")
end

function parse_dmumu_start_mode(value)
    mode = Symbol(replace(lowercase(strip(String(value))), "-" => "_"))
    mode in (:sliding, :sliding_window, :pair_start, :current) && return :sliding
    mode in (:injection, :initial, :t0, :initial_particle) && return :injection
    error("dmumu_start_mode must be sliding or injection.")
end

function dmumu_start_mode(cfg)
    return parse_dmumu_start_mode(get(cfg, :dmumu_start_mode, :sliding))
end

function mu_bin_abs(cfg)
    return Bool(get(cfg, :mu_bin_abs, true))
end

function parse_cli_config(args)
    cfg = Dict{Symbol, Any}(COMBINED_FULL_CFG)

    for argument in args
        if argument == "--help" || argument == "-h"
            print_usage()
            exit(0)
        elseif argument == "--allow-huge"
            cfg[:allow_huge] = true
        elseif argument == "--smoke"
            cfg[:n_particles_to_use] = 8
            cfg[:particle_chunk_size] = 4
            cfg[:particle_selection] = :range
            cfg[:dmumu_start_mode] = :sliding
            cfg[:lag_mode] = :uniform_samples
            cfg[:min_lag_steps] = 1
            cfg[:n_lag_samples] = 5
            cfg[:max_lag_steps] = 20
            cfg[:n_mu_bins] = 8
            cfg[:min_count_per_cell] = 2
            cfg[:field_subset] = (16, 16, 16)
            cfg[:output_dir] = joinpath(PIPELINE_ROOT, "outputs", "combined_full", "smoke")
        elseif startswith(argument, "--trajectory-h5=")
            cfg[:trajectory_h5] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--turbulence-h5=")
            cfg[:turbulence_h5] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--output-dir=")
            cfg[:output_dir] = split(argument, "=", limit=2)[2]
        elseif startswith(argument, "--compute-backend=")
            cfg[:compute_backend] = parse_backend(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--compute-precision=")
            cfg[:compute_precision] = parse_precision(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-particles=")
            cfg[:n_particles_to_use] = parse_particle_count(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--first-particle=")
            cfg[:first_particle] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-selection=")
            cfg[:particle_selection] = parse_particle_selection(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-seed=")
            cfg[:particle_seed] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-block-size=")
            cfg[:particle_block_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--particle-chunk-size=")
            cfg[:particle_chunk_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-start-mode=")
            cfg[:dmumu_start_mode] = parse_dmumu_start_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--min-lag-steps=")
            cfg[:min_lag_steps] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-min-gyroperiods=") || startswith(argument, "--dmumu-lag-min-gyroperiods=")
            cfg[:lag_min_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-lag-samples=")
            cfg[:n_lag_samples] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--max-lag-steps=")
            cfg[:max_lag_steps] = parse_maybe_int(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-max-gyroperiods=") || startswith(argument, "--dmumu-lag-max-gyroperiods=")
            cfg[:lag_max_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-mode=")
            cfg[:lag_mode] = parse_lag_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-range-policy=") || startswith(argument, "--dmumu-lag-range-policy=")
            cfg[:lag_range_policy] = normalize_lag_range_policy(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-common-scope=") || startswith(argument, "--dmumu-lag-common-scope=")
            cfg[:lag_common_scope] = normalize_lag_common_scope(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-boundary-policy=") || startswith(argument, "--dmumu-lag-boundary-policy=")
            cfg[:lag_boundary_policy] = normalize_lag_boundary_policy(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--max-lag-boundary-relative-error=") || startswith(argument, "--dmumu-max-lag-boundary-relative-error=")
            cfg[:max_lag_boundary_relative_error] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-step-stride=")
            cfg[:lag_step_stride] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-stride-gyroperiods=") || startswith(argument, "--dmumu-lag-stride-gyroperiods=")
            cfg[:lag_stride_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-mu-bins=")
            cfg[:n_mu_bins] = parse(Int, split(argument, "=", limit=2)[2])
        elseif argument == "--mu-bin-abs"
            cfg[:mu_bin_abs] = true
        elseif argument == "--no-mu-bin-abs" || argument == "--signed-mu-bins"
            cfg[:mu_bin_abs] = false
        elseif startswith(argument, "--mu-bin-abs=")
            cfg[:mu_bin_abs] = parse_bool(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--mu-min=")
            cfg[:mu_min] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--mu-max=")
            cfg[:mu_max] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--min-count-per-cell=")
            cfg[:min_count_per_cell] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--field-subset=")
            cfg[:field_subset] = parse_field_subset(split(argument, "=", limit=2)[2])
        else
            error("Unknown option: $argument. Use --help for supported options.")
        end
    end

    cfg[:output_h5] === nothing && (cfg[:output_h5] = joinpath(cfg[:output_dir], "delta_mu2_dmumu_full.h5"))
    cfg[:output_delta_png] === nothing && (cfg[:output_delta_png] = joinpath(cfg[:output_dir], "delta_mu2_curve_full.png"))
    cfg[:output_heatmap_png] === nothing && (cfg[:output_heatmap_png] = joinpath(cfg[:output_dir], "dmumu_mu_tau_full.png"))
    cfg[:output_collapsed_png] === nothing && (cfg[:output_collapsed_png] = joinpath(cfg[:output_dir], "dmumu_tau_average_full.png"))
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

function build_mu_edges_full(cfg)
    mu_minimum = Float64(cfg[:mu_min])
    mu_maximum = Float64(cfg[:mu_max])
    n_bins = Int(cfg[:n_mu_bins])
    mu_minimum < mu_maximum || error("mu_min must be less than mu_max.")
    n_bins > 0 || error("n_mu_bins must be positive.")
    if mu_bin_abs(cfg)
        mu_minimum >= 0.0 || error("mu_min must be >= 0 when mu_bin_abs is true.")
        mu_maximum <= 1.0 || error("mu_max must be <= 1 when mu_bin_abs is true.")
    end
    edges = collect(range(mu_minimum, mu_maximum, length=n_bins + 1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return edges, centers
end

@inline function mu_bin_value(mu_value::Real, bin_abs::Bool)
    return bin_abs ? abs(mu_value) : mu_value
end

function mu_bin_index_full(mu_value::Real, mu_edges::Vector{Float64})
    isfinite(mu_value) || return 0
    if mu_value < mu_edges[1] || mu_value > mu_edges[end]
        return 0
    end
    bin_index = searchsortedlast(mu_edges, Float64(mu_value))
    if bin_index == length(mu_edges)
        return length(mu_edges) - 1
    end
    return bin_index < 1 ? 0 : bin_index
end

function format_count(value::Integer)
    text = string(value)
    chunks = String[]
    while length(text) > 3
        pushfirst!(chunks, text[end-2:end])
        text = text[1:end-3]
    end
    pushfirst!(chunks, text)
    return join(chunks, "_")
end

function estimate_pair_visits(nsteps::Integer, selected_particle_count::Integer, lag_steps, start_mode::Symbol=:sliding)
    visits_per_particle = if start_mode == :injection
        length(lag_steps)
    else
        sum(nsteps - lag_step for lag_step in lag_steps)
    end
    return Int128(selected_particle_count) * Int128(visits_per_particle)
end

function guard_exact_pair_cost!(nsteps::Integer, selected_particle_count::Integer, lag_steps, cfg)
    start_mode = dmumu_start_mode(cfg)
    estimated_visits = estimate_pair_visits(nsteps, selected_particle_count, lag_steps, start_mode)
    limit = Int128(round(Int64, Float64(cfg[:max_pair_visits_without_allow])))
    println("Estimated D_mumu pair visits: ", format_count(estimated_visits))

    if estimated_visits > limit && !Bool(cfg[:allow_huge])
        error("""
        Refusing to start a very large D_mumu calculation.
        Estimated pair visits: $(format_count(estimated_visits))
        Safety limit without --allow-huge: $(format_count(limit))

        Re-run with --allow-huge if this is intentional.
        Consider first testing a smaller run with --n-particles, --max-lag-steps,
        or fewer --n-lag-samples.
        """)
    end

    return estimated_visits
end

function build_particle_indices(total_particles::Integer, cfg)
    first_particle = Int(cfg[:first_particle])
    first_particle >= 1 || error("first_particle must be >= 1.")
    first_particle <= total_particles || error("first_particle exceeds total particle count.")

    available_count = total_particles - first_particle + 1
    requested_count = cfg[:n_particles_to_use] === nothing ? available_count : Int(cfg[:n_particles_to_use])
    requested_count > 0 || error("n-particles must be positive or all.")
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

function combine_welford_stats!(counts, means, m2s, lag_index, local_count, local_mean, local_m2)
    local_count <= 0 && return nothing
    global_count = counts[lag_index]
    if global_count == 0
        counts[lag_index] = local_count
        means[lag_index] = local_mean
        m2s[lag_index] = local_m2
        return nothing
    end

    combined_count = global_count + local_count
    delta = local_mean - means[lag_index]
    means[lag_index] += delta * local_count / combined_count
    m2s[lag_index] += local_m2 + delta * delta * global_count * local_count / combined_count
    counts[lag_index] = combined_count
    return nothing
end

function process_mu_chunk_full_pairs_cpu!(
    mu_chunk,
    lag_steps::Vector{Int},
    mu_edges::Vector{Float64},
    particle_counts,
    particle_means,
    particle_m2s,
    pair_sum_squares,
    pair_counts,
    dmumu_counts,
    dmumu_sum_delta,
    dmumu_sum_delta2,
    bin_abs::Bool,
)
    nsteps, chunk_particle_count = size(mu_chunk)
    n_bins = length(mu_edges) - 1
    thread_count = Threads.maxthreadid()

    local_dmumu_counts = [zeros(Int64, n_bins) for _ in 1:thread_count]
    local_dmumu_sum_delta = [zeros(Float64, n_bins) for _ in 1:thread_count]
    local_dmumu_sum_delta2 = [zeros(Float64, n_bins) for _ in 1:thread_count]
    local_particle_counts = zeros(Int64, thread_count)
    local_particle_means = zeros(Float64, thread_count)
    local_particle_m2s = zeros(Float64, thread_count)
    local_pair_sum_squares = zeros(Float64, thread_count)
    local_pair_counts = zeros(Int64, thread_count)

    @inbounds for (lag_index, lag_step) in enumerate(lag_steps)
        lag_step >= nsteps && continue
        last_start = nsteps - lag_step

        for thread_index in 1:thread_count
            fill!(local_dmumu_counts[thread_index], 0)
            fill!(local_dmumu_sum_delta[thread_index], 0.0)
            fill!(local_dmumu_sum_delta2[thread_index], 0.0)
        end
        fill!(local_particle_counts, 0)
        fill!(local_particle_means, 0.0)
        fill!(local_particle_m2s, 0.0)
        fill!(local_pair_sum_squares, 0.0)
        fill!(local_pair_counts, 0)

        Threads.@threads for particle_index in 1:chunk_particle_count
            thread_index = Threads.threadid()
            bin_counts = local_dmumu_counts[thread_index]
            bin_sum_delta = local_dmumu_sum_delta[thread_index]
            bin_sum_delta2 = local_dmumu_sum_delta2[thread_index]

            particle_sumsq = 0.0
            valid_pair_count = 0

            for step_index in 1:last_start
                mu_start = Float64(mu_chunk[step_index, particle_index])
                mu_end = Float64(mu_chunk[step_index + lag_step, particle_index])
                if !(isfinite(mu_start) && isfinite(mu_end))
                    continue
                end

                delta_mu = mu_end - mu_start
                delta_mu2 = delta_mu * delta_mu
                particle_sumsq += delta_mu2
                valid_pair_count += 1

                bin_index = mu_bin_index_full(mu_bin_value(mu_start, bin_abs), mu_edges)
                if 1 <= bin_index <= n_bins
                    bin_counts[bin_index] += 1
                    bin_sum_delta[bin_index] += delta_mu
                    bin_sum_delta2[bin_index] += delta_mu2
                end
            end

            if valid_pair_count > 0
                particle_value = particle_sumsq / valid_pair_count
                local_particle_counts[thread_index] += 1
                delta = particle_value - local_particle_means[thread_index]
                local_particle_means[thread_index] += delta / local_particle_counts[thread_index]
                delta2 = particle_value - local_particle_means[thread_index]
                local_particle_m2s[thread_index] += delta * delta2
                local_pair_sum_squares[thread_index] += particle_sumsq
                local_pair_counts[thread_index] += valid_pair_count
            end
        end

        for thread_index in 1:thread_count
            dmumu_counts[:, lag_index] .+= local_dmumu_counts[thread_index]
            dmumu_sum_delta[:, lag_index] .+= local_dmumu_sum_delta[thread_index]
            dmumu_sum_delta2[:, lag_index] .+= local_dmumu_sum_delta2[thread_index]

            combine_welford_stats!(
                particle_counts,
                particle_means,
                particle_m2s,
                lag_index,
                Int(local_particle_counts[thread_index]),
                local_particle_means[thread_index],
                local_particle_m2s[thread_index],
            )
            pair_sum_squares[lag_index] += local_pair_sum_squares[thread_index]
            pair_counts[lag_index] += local_pair_counts[thread_index]
        end
    end

    return nothing
end

function process_mu_chunk_injection_pairs_cpu!(
    mu_chunk,
    lag_steps::Vector{Int},
    mu_edges::Vector{Float64},
    particle_counts,
    particle_means,
    particle_m2s,
    pair_sum_squares,
    pair_counts,
    dmumu_counts,
    dmumu_sum_delta,
    dmumu_sum_delta2,
    bin_abs::Bool,
)
    nsteps, chunk_particle_count = size(mu_chunk)
    n_bins = length(mu_edges) - 1
    thread_count = Threads.maxthreadid()

    local_dmumu_counts = [zeros(Int64, n_bins) for _ in 1:thread_count]
    local_dmumu_sum_delta = [zeros(Float64, n_bins) for _ in 1:thread_count]
    local_dmumu_sum_delta2 = [zeros(Float64, n_bins) for _ in 1:thread_count]
    local_particle_counts = zeros(Int64, thread_count)
    local_particle_means = zeros(Float64, thread_count)
    local_particle_m2s = zeros(Float64, thread_count)
    local_pair_sum_squares = zeros(Float64, thread_count)
    local_pair_counts = zeros(Int64, thread_count)

    @inbounds for (lag_index, lag_step) in enumerate(lag_steps)
        lag_step >= nsteps && continue

        for thread_index in 1:thread_count
            fill!(local_dmumu_counts[thread_index], 0)
            fill!(local_dmumu_sum_delta[thread_index], 0.0)
            fill!(local_dmumu_sum_delta2[thread_index], 0.0)
        end
        fill!(local_particle_counts, 0)
        fill!(local_particle_means, 0.0)
        fill!(local_particle_m2s, 0.0)
        fill!(local_pair_sum_squares, 0.0)
        fill!(local_pair_counts, 0)

        Threads.@threads for particle_index in 1:chunk_particle_count
            thread_index = Threads.threadid()
            bin_counts = local_dmumu_counts[thread_index]
            bin_sum_delta = local_dmumu_sum_delta[thread_index]
            bin_sum_delta2 = local_dmumu_sum_delta2[thread_index]

            mu_start = Float64(mu_chunk[1, particle_index])
            mu_end = Float64(mu_chunk[1 + lag_step, particle_index])
            if !(isfinite(mu_start) && isfinite(mu_end))
                continue
            end

            delta_mu = mu_end - mu_start
            delta_mu2 = delta_mu * delta_mu

            bin_index = mu_bin_index_full(mu_bin_value(mu_start, bin_abs), mu_edges)
            if 1 <= bin_index <= n_bins
                bin_counts[bin_index] += 1
                bin_sum_delta[bin_index] += delta_mu
                bin_sum_delta2[bin_index] += delta_mu2
            end

            local_particle_counts[thread_index] += 1
            delta = delta_mu2 - local_particle_means[thread_index]
            local_particle_means[thread_index] += delta / local_particle_counts[thread_index]
            delta2 = delta_mu2 - local_particle_means[thread_index]
            local_particle_m2s[thread_index] += delta * delta2
            local_pair_sum_squares[thread_index] += delta_mu2
            local_pair_counts[thread_index] += 1
        end

        for thread_index in 1:thread_count
            dmumu_counts[:, lag_index] .+= local_dmumu_counts[thread_index]
            dmumu_sum_delta[:, lag_index] .+= local_dmumu_sum_delta[thread_index]
            dmumu_sum_delta2[:, lag_index] .+= local_dmumu_sum_delta2[thread_index]

            combine_welford_stats!(
                particle_counts,
                particle_means,
                particle_m2s,
                lag_index,
                Int(local_particle_counts[thread_index]),
                local_particle_means[thread_index],
                local_particle_m2s[thread_index],
            )
            pair_sum_squares[lag_index] += local_pair_sum_squares[thread_index]
            pair_counts[lag_index] += local_pair_counts[thread_index]
        end
    end

    return nothing
end

function process_mu_chunk_dmumu_cpu!(
    mu_chunk,
    lag_steps::Vector{Int},
    mu_edges::Vector{Float64},
    particle_counts,
    particle_means,
    particle_m2s,
    pair_sum_squares,
    pair_counts,
    dmumu_counts,
    dmumu_sum_delta,
    dmumu_sum_delta2,
    start_mode::Symbol,
    bin_abs::Bool,
)
    if start_mode == :injection
        return process_mu_chunk_injection_pairs_cpu!(
            mu_chunk,
            lag_steps,
            mu_edges,
            particle_counts,
            particle_means,
            particle_m2s,
            pair_sum_squares,
            pair_counts,
            dmumu_counts,
            dmumu_sum_delta,
            dmumu_sum_delta2,
            bin_abs,
        )
    elseif start_mode == :sliding
        return process_mu_chunk_full_pairs_cpu!(
            mu_chunk,
            lag_steps,
            mu_edges,
            particle_counts,
            particle_means,
            particle_m2s,
            pair_sum_squares,
            pair_counts,
            dmumu_counts,
            dmumu_sum_delta,
            dmumu_sum_delta2,
            bin_abs,
        )
    end

    error("Unknown dmumu_start_mode: $(start_mode)")
end

@inline function gpu_mu_bin_index(mu_value, mu_minimum, mu_maximum, inv_bin_width, n_bins::Int32, bin_abs::Bool)
    value = bin_abs ? abs(mu_value) : mu_value
    isfinite(value) || return Int32(0)
    if value < mu_minimum || value > mu_maximum
        return Int32(0)
    end
    raw = floor(Int32, (value - mu_minimum) * inv_bin_width) + Int32(1)
    if raw == n_bins + Int32(1) && value == mu_maximum
        return n_bins
    end
    if raw < Int32(1) || raw > n_bins
        return Int32(0)
    end
    return raw
end

function dmumu_particle_partials_kernel!(partial_counts, partial_sum_delta, partial_sum_delta2, particle_pair_counts, particle_sumsq_output, mu, lag_steps, mu_minimum, mu_maximum, inv_bin_width, n_bins::Int32, start_mode_code::Int32, bin_abs::Bool)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    nsteps = size(mu, 1)
    n_particles = size(mu, 2)
    n_lags = length(lag_steps)
    total = n_particles * n_lags
    T = eltype(mu)

    while index <= total
        lag_index = ((index - 1) % n_lags) + 1
        particle = ((index - 1) ÷ n_lags) + 1
        lag_step = Int(lag_steps[lag_index])

        local_particle_sumsq = zero(T)
        valid_pair_count = Int32(0)
        if lag_step < nsteps
            if start_mode_code == Int32(1)
                mu_start = mu[1, particle]
                mu_end = mu[1 + lag_step, particle]
                if isfinite(mu_start) && isfinite(mu_end)
                    delta_mu = mu_end - mu_start
                    delta_mu2 = delta_mu * delta_mu
                    bin_index = gpu_mu_bin_index(mu_start, T(mu_minimum), T(mu_maximum), T(inv_bin_width), n_bins, bin_abs)
                    if Int32(1) <= bin_index <= n_bins
                        partial_counts[bin_index, lag_index, particle] += Int32(1)
                        partial_sum_delta[bin_index, lag_index, particle] += delta_mu
                        partial_sum_delta2[bin_index, lag_index, particle] += delta_mu2
                    end
                    local_particle_sumsq = delta_mu2
                    valid_pair_count = Int32(1)
                end
            else
                last_start = nsteps - lag_step
                @inbounds for step_index in 1:last_start
                    mu_start = mu[step_index, particle]
                    mu_end = mu[step_index + lag_step, particle]
                    if isfinite(mu_start) && isfinite(mu_end)
                        delta_mu = mu_end - mu_start
                        delta_mu2 = delta_mu * delta_mu
                        bin_index = gpu_mu_bin_index(mu_start, T(mu_minimum), T(mu_maximum), T(inv_bin_width), n_bins, bin_abs)
                        if Int32(1) <= bin_index <= n_bins
                            partial_counts[bin_index, lag_index, particle] += Int32(1)
                            partial_sum_delta[bin_index, lag_index, particle] += delta_mu
                            partial_sum_delta2[bin_index, lag_index, particle] += delta_mu2
                        end
                        local_particle_sumsq += delta_mu2
                        valid_pair_count += Int32(1)
                    end
                end
            end
        end

        if valid_pair_count > 0
            particle_pair_counts[lag_index, particle] = valid_pair_count
            particle_sumsq_output[lag_index, particle] = local_particle_sumsq
        end
        index += stride
    end
    return
end

@inline function welford_merge_count_mean_m2(n_a::Int64, mean_a, m2_a, n_b::Int64, mean_b, m2_b)
    n_b <= 0 && return n_a, mean_a, m2_a
    n_a <= 0 && return n_b, mean_b, m2_b
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (typeof(mean_a)(n_b) / typeof(mean_a)(n))
    m2 = m2_a + m2_b + delta * delta * (typeof(mean_a)(n_a) * typeof(mean_a)(n_b) / typeof(mean_a)(n))
    return n, mean, m2
end

function dmumu_reduce_partials_kernel!(campaign_counts, campaign_sum_delta, campaign_sum_delta2, campaign_pair_sum_squares, campaign_pair_counts, partial_counts, partial_sum_delta, partial_sum_delta2, particle_pair_counts, particle_sumsq, lag_offset::Int32)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    n_bins = size(partial_counts, 1)
    n_lags = size(partial_counts, 2)
    n_particles = size(partial_counts, 3)
    total_hist = n_bins * n_lags * n_particles
    while index <= total_hist
        bin_index = ((index - 1) % n_bins) + 1
        local_lag = (((index - 1) ÷ n_bins) % n_lags) + 1
        particle = ((index - 1) ÷ (n_bins * n_lags)) + 1
        count = partial_counts[bin_index, local_lag, particle]
        if count > Int32(0)
            lag_index = Int(lag_offset) + local_lag - 1
            CUDA.@atomic campaign_counts[bin_index, lag_index] += Int64(count)
            CUDA.@atomic campaign_sum_delta[bin_index, lag_index] += partial_sum_delta[bin_index, local_lag, particle]
            CUDA.@atomic campaign_sum_delta2[bin_index, lag_index] += partial_sum_delta2[bin_index, local_lag, particle]
        end
        index += stride
    end

    return
end

function dmumu_reduce_particle_welford_kernel!(campaign_particle_counts, campaign_particle_means, campaign_particle_m2s, campaign_pair_sum_squares, campaign_pair_counts, particle_pair_counts, particle_sumsq, lag_offset::Int32)
    local_lag = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_lags = size(particle_pair_counts, 1)
    local_lag > n_lags && return
    n_particles = size(particle_pair_counts, 2)
    T = eltype(campaign_particle_means)
    local_count = Int64(0)
    local_mean = zero(T)
    local_m2 = zero(T)
    local_pair_sumsq = zero(T)
    local_pair_count = Int64(0)
    @inbounds for particle in 1:n_particles
        pair_count = particle_pair_counts[local_lag, particle]
        if pair_count > Int32(0)
            value = T(particle_sumsq[local_lag, particle]) / T(pair_count)
            local_count, local_mean, local_m2 = welford_merge_count_mean_m2(local_count, local_mean, local_m2, Int64(1), value, zero(T))
            local_pair_sumsq += T(particle_sumsq[local_lag, particle])
            local_pair_count += Int64(pair_count)
        end
    end
    if local_count > 0
        lag_index = Int(lag_offset) + local_lag - 1
        old_count = campaign_particle_counts[lag_index]
        old_mean = campaign_particle_means[lag_index]
        old_m2 = campaign_particle_m2s[lag_index]
        new_count, new_mean, new_m2 = welford_merge_count_mean_m2(old_count, old_mean, old_m2, local_count, local_mean, local_m2)
        campaign_particle_counts[lag_index] = new_count
        campaign_particle_means[lag_index] = new_mean
        campaign_particle_m2s[lag_index] = new_m2
        campaign_pair_sum_squares[lag_index] += local_pair_sumsq
        campaign_pair_counts[lag_index] += local_pair_count
    end
    return
end

function process_mu_chunk_dmumu_gpu!(dmu_chunk, lag_steps::Vector{Int}, lag_range, mu_edges::Vector{Float64}, gpu_accumulators, start_mode::Symbol, bin_abs::Bool, cfg)
    lag_subset = lag_steps[lag_range]
    n_lags = length(lag_subset)
    n_bins = length(mu_edges) - 1
    n_particles = size(dmu_chunk, 2)
    dlag_steps = CuArray(Int32.(lag_subset))
    T = eltype(dmu_chunk)
    partial_counts = CUDA.zeros(Int32, n_bins, n_lags, n_particles)
    partial_sum_delta = CUDA.zeros(T, n_bins, n_lags, n_particles)
    partial_sum_delta2 = CUDA.zeros(T, n_bins, n_lags, n_particles)
    particle_pair_counts = CUDA.zeros(Int32, n_lags, n_particles)
    particle_sumsq = CUDA.zeros(T, n_lags, n_particles)

    threads = Int(get(cfg, :gpu_threads, 256))
    blocks = min(4096, cld(n_particles * n_lags, threads))
    start_mode_code = start_mode == :injection ? Int32(1) : Int32(0)
    inv_bin_width = n_bins / (mu_edges[end] - mu_edges[1])
    @cuda threads=threads blocks=blocks dmumu_particle_partials_kernel!(
        partial_counts,
        partial_sum_delta,
        partial_sum_delta2,
        particle_pair_counts,
        particle_sumsq,
        dmu_chunk,
        dlag_steps,
        Float64(mu_edges[1]),
        Float64(mu_edges[end]),
        Float64(inv_bin_width),
        Int32(n_bins),
        start_mode_code,
        bin_abs,
    )
    reduce_total = max(length(partial_counts), length(particle_pair_counts))
    reduce_blocks = min(4096, cld(reduce_total, threads))
    @cuda threads=threads blocks=reduce_blocks dmumu_reduce_partials_kernel!(
        gpu_accumulators.counts,
        gpu_accumulators.sum_delta,
        gpu_accumulators.sum_delta2,
        gpu_accumulators.pair_sum_squares,
        gpu_accumulators.pair_counts,
        partial_counts,
        partial_sum_delta,
        partial_sum_delta2,
        particle_pair_counts,
        particle_sumsq,
        Int32(first(lag_range)),
    )
    @cuda threads=threads blocks=cld(n_lags, threads) dmumu_reduce_particle_welford_kernel!(
        gpu_accumulators.particle_counts,
        gpu_accumulators.particle_means,
        gpu_accumulators.particle_m2s,
        gpu_accumulators.pair_sum_squares,
        gpu_accumulators.pair_counts,
        particle_pair_counts,
        particle_sumsq,
        Int32(first(lag_range)),
    )
    return nothing
end

function create_dmumu_gpu_accumulators(n_bins::Integer, n_lags::Integer)
    return (
        counts = CUDA.zeros(Int64, n_bins, n_lags),
        sum_delta = CUDA.zeros(Float32, n_bins, n_lags),
        sum_delta2 = CUDA.zeros(Float32, n_bins, n_lags),
        particle_counts = CUDA.zeros(Int64, n_lags),
        particle_means = CUDA.zeros(Float32, n_lags),
        particle_m2s = CUDA.zeros(Float32, n_lags),
        pair_sum_squares = CUDA.zeros(Float32, n_lags),
        pair_counts = CUDA.zeros(Int64, n_lags),
    )
end

function copy_dmumu_gpu_accumulators!(gpu_accumulators, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts, dmumu_counts, dmumu_sum_delta, dmumu_sum_delta2)
    dmumu_counts .= Array(gpu_accumulators.counts)
    dmumu_sum_delta .= Float64.(Array(gpu_accumulators.sum_delta))
    dmumu_sum_delta2 .= Float64.(Array(gpu_accumulators.sum_delta2))
    particle_count_values = Array(gpu_accumulators.particle_counts)
    particle_mean_values = Array(gpu_accumulators.particle_means)
    particle_m2_values = Array(gpu_accumulators.particle_m2s)
    pair_sum_squares .= Float64.(Array(gpu_accumulators.pair_sum_squares))
    pair_counts .= Array(gpu_accumulators.pair_counts)
    @inbounds for lag_index in eachindex(particle_counts)
        count = particle_count_values[lag_index]
        particle_counts[lag_index] = Int(count)
        if count > 0
            particle_means[lag_index] = Float64(particle_mean_values[lag_index])
            particle_m2s[lag_index] = max(0.0, Float64(particle_m2_values[lag_index]))
        end
    end
    return nothing
end

function compute_dmumu_arrays_full(counts, sum_delta, sum_delta2, tau_s, tau_norm, cfg)
    n_bins, n_lags = size(counts)
    min_count = Int(cfg[:min_count_per_cell])
    mean_delta = fill(NaN, n_bins, n_lags)
    mean_delta2 = fill(NaN, n_bins, n_lags)
    drift_norm = fill(NaN, n_bins, n_lags)
    dmumu_raw_norm = fill(NaN, n_bins, n_lags)
    dmumu_centered_norm = fill(NaN, n_bins, n_lags)
    dmumu_raw_per_s = fill(NaN, n_bins, n_lags)
    dmumu_centered_per_s = fill(NaN, n_bins, n_lags)

    @inbounds for lag_index in 1:n_lags
        normalized_lag = Float64(tau_norm[lag_index])
        physical_lag = Float64(tau_s[lag_index])
        for bin_index in 1:n_bins
            count = counts[bin_index, lag_index]
            count >= min_count || continue
            delta_mean = sum_delta[bin_index, lag_index] / count
            delta2_mean = sum_delta2[bin_index, lag_index] / count
            centered_delta2 = max(0.0, delta2_mean - delta_mean * delta_mean)

            mean_delta[bin_index, lag_index] = delta_mean
            mean_delta2[bin_index, lag_index] = delta2_mean
            drift_norm[bin_index, lag_index] = delta_mean / normalized_lag
            dmumu_raw_norm[bin_index, lag_index] = delta2_mean / (2.0 * normalized_lag)
            dmumu_centered_norm[bin_index, lag_index] = centered_delta2 / (2.0 * normalized_lag)
            dmumu_raw_per_s[bin_index, lag_index] = delta2_mean / (2.0 * physical_lag)
            dmumu_centered_per_s[bin_index, lag_index] = centered_delta2 / (2.0 * physical_lag)
        end
    end

    return mean_delta,
           mean_delta2,
           drift_norm,
           dmumu_raw_norm,
           dmumu_centered_norm,
           dmumu_raw_per_s,
           dmumu_centered_per_s
end

function average_over_tau_full(values, counts, cfg)
    n_bins, n_lags = size(values)
    min_count = Int(cfg[:min_count_per_cell])
    unweighted = fill(NaN, n_bins)
    count_weighted = fill(NaN, n_bins)

    @inbounds for bin_index in 1:n_bins
        value_sum = 0.0
        valid_lags = 0
        weighted_sum = 0.0
        total_weight = 0
        for lag_index in 1:n_lags
            value = values[bin_index, lag_index]
            count = counts[bin_index, lag_index]
            if isfinite(value) && count >= min_count
                value_sum += value
                valid_lags += 1
                weighted_sum += value * count
                total_weight += count
            end
        end
        if valid_lags > 0
            unweighted[bin_index] = value_sum / valid_lags
        end
        if total_weight > 0
            count_weighted[bin_index] = weighted_sum / total_weight
        end
    end

    return unweighted, count_weighted
end

function mu_bin_axis_label(start_mode::Symbol, bin_abs::Bool)
    if bin_abs
        return start_mode == :injection ? raw"$|\mu_0|$" : raw"$|\mu(t)|$"
    end
    return start_mode == :injection ? raw"$\mu_0$" : raw"$\mu(t)$"
end

function dmumu_plot_title(bin_abs::Bool)
    return bin_abs ? raw"$D_{\mu\mu}(|\mu|,\tau)$" : raw"$D_{\mu\mu}(\mu,\tau)$"
end

function dmumu_tau_average_title(bin_abs::Bool)
    return bin_abs ? raw"Tau-averaged $D_{\mu\mu}(|\mu|)$" : raw"Tau-averaged $D_{\mu\mu}(\mu)$"
end

function mu_bin_coordinate_name(start_mode::Symbol, bin_abs::Bool)
    base = start_mode == :injection ? "mu(0)" : "mu(t)"
    return bin_abs ? "abs(" * base * ")" : base
end

function save_combined_full_h5(
    path_h5::AbstractString,
    cfg,
    backend::Symbol,
    estimated_pair_visits,
    delta_df,
    mu_edges,
    mu_centers,
    lag_steps,
    tau_s,
    tau_norm,
    first_particle,
    last_particle,
    particle_indices,
    dmumu_counts,
    dmumu_sum_delta,
    dmumu_sum_delta2,
    mean_delta,
    mean_delta2,
    drift_norm,
    dmumu_raw_norm,
    dmumu_centered_norm,
    dmumu_raw_per_s,
    dmumu_centered_per_s,
    collapsed_raw_norm,
    collapsed_centered_norm,
    collapsed_raw_norm_count_weighted,
    collapsed_centered_norm_count_weighted
    ; lag_grid=nothing,
)
    mkpath(dirname(path_h5))
    h5open(path_h5, "w") do file
        compute_dmumu = delta_df !== nothing && dmumu_counts !== nothing
        file["trajectory_h5"] = string(cfg[:trajectory_h5])
        file["turbulence_h5"] = string(cfg[:turbulence_h5])
        file["compute_backend"] = string(backend)
        file["requested_compute_backend"] = string(get(cfg, :compute_backend, backend))
        file["resolved_compute_backend"] = string(backend)
        file["compute_precision"] = string(cfg[:compute_precision])
        file["accumulator_precision"] = "Float32"
        file["backend_version"] = backend == :gpu ? "gpu_dmumu_partial_reduce_v2" : "cpu_dmumu_reference_v1"
        file["gpu_lag_batch_size"] = Int(get(cfg, :resolved_gpu_lag_batch_size, get(cfg, :gpu_lag_batch_size, 1)))
        file["requested_gpu_lag_batch_size"] = Int(get(cfg, :gpu_lag_batch_size, 1))
        file["resolved_particle_chunk_size"] = Int(get(cfg, :resolved_particle_chunk_size, cfg[:particle_chunk_size]))
        file["estimated_peak_gpu_memory"] = Int64(get(cfg, :estimated_peak_gpu_memory, 0))
        file["usable_gpu_memory"] = Int64(get(cfg, :usable_gpu_memory, 0))
        file["memory_planner_fallback_used"] = Bool(get(cfg, :memory_planner_fallback_used, false))
        file["gpu_memory_fraction"] = Float64(get(cfg, :gpu_memory_fraction, 1.0))
        file["postprocess_pipeline_enabled"] = false
        file["postprocess_pipeline_buffer_count"] = Int(get(cfg, :gpu_pipeline_buffers, 1))
        file["postprocess_async_h2d_enabled"] = false
        file["postprocess_transfer_stream_enabled"] = false
        file["postprocess_compute_stream_enabled"] = false
        file["source_cache_uniform_time_axis"] = true
        file["source_cache_identity"] = string(cfg[:trajectory_h5])
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
        file["min_count_per_cell"] = Int(cfg[:min_count_per_cell])
        file["box_length_pc"] = Float64(cfg[:box_length_pc])
        file["field_subset"] = cfg[:field_subset] === nothing ? "nothing" : string(cfg[:field_subset])
        file["estimated_pair_visits"] = string(estimated_pair_visits)
        file["tau_average_note"] = "Lag average over the selected tau grid; this is an effective lag-averaged scattering measure, not a fitted diffusive plateau."
        if compute_dmumu
            start_mode = dmumu_start_mode(cfg)
            bin_abs = mu_bin_abs(cfg)
            file["dmumu_start_mode"] = string(start_mode)
            file["mu_bin_abs"] = bin_abs
            file["mu_bin_coordinate"] = mu_bin_coordinate_name(start_mode, bin_abs)
            file["n_mu_bins"] = Int(cfg[:n_mu_bins])
            start_note = start_mode == :injection ?
                "Injection-anchored accumulation: each selected particle contributes at most one pair per selected lag, Delta mu = mu(tau) - mu(0), binned by injected mu(0)." :
                "Sliding full-pair accumulation over every selected particle and every valid start time for each selected lag; Delta mu = mu(t + tau) - mu(t), binned by pair-start mu(t)."
            bin_note = bin_abs ?
                " D_mumu bins use the absolute value of the start mu; stored mu values and Delta mu remain signed." :
                " D_mumu bins use the signed start mu."
            file["estimator_note"] = start_note * bin_note * " No random pair sampling."

            delta_group = create_group(file, "delta_mu2")
            for column_name in names(delta_df)
                delta_group[string(column_name)] = collect(delta_df[!, column_name])
            end
            if lag_grid !== nothing
                delta_group["requested_tau_gyroperiods"] = lag_grid.requested_tau_gyroperiods
                delta_group["common_requested_tau_gyroperiods"] = lag_grid.common_requested_tau_gyroperiods
                delta_group["lag_mapping_error_gyroperiods"] = lag_grid.lag_mapping_error_gyroperiods
            end

            dmumu_group = create_group(file, "dmumu")
            dmumu_group["mu_edges"] = mu_edges
            dmumu_group["mu_centers"] = mu_centers
            dmumu_group["lag_step"] = lag_steps
            dmumu_group["lag_steps"] = lag_steps
            if lag_grid !== nothing
                dmumu_group["requested_tau_gyroperiods"] = lag_grid.requested_tau_gyroperiods
                dmumu_group["common_requested_tau_gyroperiods"] = lag_grid.common_requested_tau_gyroperiods
                dmumu_group["tau_gyroperiods"] = lag_grid.tau_gyroperiods
                dmumu_group["lag_mapping_error_gyroperiods"] = lag_grid.lag_mapping_error_gyroperiods
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
                dmumu_group["requested_tau_gyroperiods"] = tOmega0_to_gyroperiods.(tau_norm)
                dmumu_group["tau_gyroperiods"] = tOmega0_to_gyroperiods.(tau_norm)
                dmumu_group["lag_mapping_error_gyroperiods"] = zeros(Float64, length(lag_steps))
            end
            dmumu_group["tau_s"] = tau_s
            dmumu_group["tau_norm"] = tau_norm
            dmumu_group["count_pairs_full"] = dmumu_counts
            dmumu_group["sum_delta_mu"] = dmumu_sum_delta
            dmumu_group["sum_delta_mu2"] = dmumu_sum_delta2
            dmumu_group["mean_delta_mu"] = mean_delta
            dmumu_group["mean_delta_mu2"] = mean_delta2
            dmumu_group["A_mu_norm"] = drift_norm
            dmumu_group["D_mumu_raw_norm"] = dmumu_raw_norm
            dmumu_group["D_mumu_centered_norm"] = dmumu_centered_norm
            dmumu_group["D_mumu_raw_per_s"] = dmumu_raw_per_s
            dmumu_group["D_mumu_centered_per_s"] = dmumu_centered_per_s
            dmumu_group["D_mumu_raw_tau_average_norm"] = collapsed_raw_norm
            dmumu_group["D_mumu_centered_tau_average_norm"] = collapsed_centered_norm
            dmumu_group["D_mumu_raw_tau_average_norm_count_weighted"] = collapsed_raw_norm_count_weighted
            dmumu_group["D_mumu_centered_tau_average_norm_count_weighted"] = collapsed_centered_norm_count_weighted
        end

    end

    return nothing
end

function plot_dmumu_heatmap_full(
    path_png::AbstractString,
    mu_edges,
    tau_gyroperiods,
    dmumu_centered_norm;
    use_usetex::Bool=false,
    mu_axis_label::AbstractString=raw"$\mu_0$",
    title_label::AbstractString=raw"$D_{\mu\mu}(\mu,\tau)$",
)
    mkpath(dirname(path_png))
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(8, 4.8))
    imshow(
        dmumu_centered_norm,
        aspect="auto",
        origin="lower",
        extent=(minimum(tau_gyroperiods), maximum(tau_gyroperiods), mu_edges[1], mu_edges[end]),
        cmap="viridis",
    )
    colorbar(label=raw"$D_{\mu\mu}/\Omega_0$ (centered)")
    xlabel("Lag [reference gyroperiods]")
    ylabel(mu_axis_label)
    title(title_label)
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
    return nothing
end

function plot_collapsed_dmumu_full(
    path_png::AbstractString,
    mu_edges,
    mu_centers,
    collapsed_raw_norm,
    collapsed_centered_norm,
    collapsed_raw_norm_count_weighted,
    collapsed_centered_norm_count_weighted;
    use_usetex::Bool=false,
    mu_axis_label::AbstractString=raw"$\mu_0$",
    title_label::AbstractString=raw"Tau-averaged $D_{\mu\mu}(\mu)$",
)
    mkpath(dirname(path_png))
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(7, 4))
    plot(mu_centers, collapsed_centered_norm, "o-", color="black", linewidth=1.5, markersize=4, label="centered, tau-average")
    plot(mu_centers, collapsed_raw_norm, "s--", color="tab:blue", linewidth=1.2, markersize=3, label="raw, tau-average")
    plot(mu_centers, collapsed_centered_norm_count_weighted, ":", color="0.35", linewidth=1.4, label="centered, count-weighted")
    plot(mu_centers, collapsed_raw_norm_count_weighted, "-.", color="tab:orange", linewidth=1.1, label="raw, count-weighted")
    xlim(mu_edges[1], mu_edges[end])
    xlabel(mu_axis_label)
    ylabel(raw"$\langle D_{\mu\mu}/\Omega_0 \rangle_\tau$")
    title(title_label)
    grid(true, alpha=0.3)
    legend(frameon=false, fontsize=8)
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
    return nothing
end

function run_combined_full(cfg)
    mkpath(cfg[:output_dir])
    mkpath(dirname(cfg[:output_h5]))
    compute_dmumu = Bool(get(cfg, :compute_dmumu, true))
    compute_dmumu && mkpath(dirname(cfg[:output_delta_png]))
    compute_dmumu && mkpath(dirname(cfg[:output_heatmap_png]))
    compute_dmumu && mkpath(dirname(cfg[:output_collapsed_png]))
    if !compute_dmumu
        rm(cfg[:output_delta_png]; force=true)
        rm(cfg[:output_heatmap_png]; force=true)
        rm(cfg[:output_collapsed_png]; force=true)
    end
    compute_dmumu || error("D_mumu analysis is disabled.")

    backend = compute_dmumu ? resolve_backend(cfg) : :not_used
    T = cfg[:compute_precision]
    mu_edges = Float64[]
    mu_centers = Float64[]
    start_mode = dmumu_start_mode(cfg)
    bin_abs = mu_bin_abs(cfg)
    mu_axis_label = ""
    heatmap_title = ""
    collapsed_title = ""
    Bx = nothing
    By = nothing
    Bz = nothing
    xgrid = nothing
    ygrid = nothing
    zgrid = nothing

    if compute_dmumu
        mu_edges, mu_centers = build_mu_edges_full(cfg)
        mu_axis_label = mu_bin_axis_label(start_mode, bin_abs)
        heatmap_title = dmumu_plot_title(bin_abs)
        collapsed_title = dmumu_tau_average_title(bin_abs)

        println("Loading magnetic field from ", cfg[:turbulence_h5])
        Bx, By, Bz = load_B_fields(cfg, T)
        nx, ny, nz = size(Bx)
        xgrid, ygrid, zgrid = build_uniform_coords(cfg, nx, ny, nz, T)
    end

    dBx = nothing
    dBy = nothing
    dBz = nothing
    if compute_dmumu && backend == :gpu
        println("Moving magnetic field to GPU")
        dBx = CuArray(Bx)
        dBy = CuArray(By)
        dBz = CuArray(Bz)
    end

    h5open(cfg[:trajectory_h5], "r") do trajectory_file
        positions_dataset = trajectory_file["positions"]
        momenta_dataset = trajectory_file["momenta"]
        t_s = Float64.(read(trajectory_file["t_s"]))
        t_norm = Float64.(read(trajectory_file["t_norm"]))
        t_gyroperiods = haskey(trajectory_file, "t_gyroperiods") ? Float64.(read(trajectory_file["t_gyroperiods"])) : t_gyroperiods_from_axes(t_s, t_norm)
        nsteps = validate_trajectory_layout(positions_dataset, momenta_dataset, t_s, t_norm)
        validate_time_axes(t_s, t_norm, t_gyroperiods; key_name="Trajectory HDF5 time axes in " * string(cfg[:trajectory_h5]), require_uniform=true)
        total_particles = size(positions_dataset, 1)

        particle_indices = build_particle_indices(total_particles, cfg)
        first_particle = first(particle_indices)
        last_particle = last(particle_indices)
        selected_particle_count = length(particle_indices)
        selected_particle_count > 0 || error("No particles selected.")

        lag_grid = resolve_lag_grid(cfg, t_gyroperiods)
        if lag_grid.duplicate_lag_mapping_count > 0 && lag_grid.duplicate_lag_mapping_count / lag_grid.requested_lag_count > 0.10
            @warn "Requested D_mumu lag grid collapsed substantially after mapping to cached-step offsets." requested=lag_grid.requested_lag_count unique=lag_grid.unique_lag_count duplicates=lag_grid.duplicate_lag_mapping_count
        end
        lag_steps = lag_grid.lag_steps
        tau_s = [Float64(t_s[lag_step + 1] - t_s[1]) for lag_step in lag_steps]
        tau_norm = Float64.(lag_grid.tau_norm)
        tau_gyroperiods = Float64.(lag_grid.tau_gyroperiods)
        estimated_pair_visits = guard_exact_pair_cost!(nsteps, selected_particle_count, lag_steps, cfg)

        n_lags = length(lag_steps)
        n_bins = length(mu_centers)
        particle_counts = compute_dmumu ? zeros(Int, n_lags) : nothing
        particle_means = compute_dmumu ? zeros(Float64, n_lags) : nothing
        particle_m2s = compute_dmumu ? zeros(Float64, n_lags) : nothing
        pair_sum_squares = compute_dmumu ? zeros(Float64, n_lags) : nothing
        pair_counts = compute_dmumu ? zeros(Int64, n_lags) : nothing
        dmumu_counts = compute_dmumu ? zeros(Int64, n_bins, n_lags) : nothing
        dmumu_sum_delta = compute_dmumu ? zeros(Float64, n_bins, n_lags) : nothing
        dmumu_sum_delta2 = compute_dmumu ? zeros(Float64, n_bins, n_lags) : nothing
        memory_plan = backend == :gpu ? resolve_gpu_work_plan(
            cfg;
            particle_chunk_size=min(Int(cfg[:particle_chunk_size]), selected_particle_count),
            lag_count=n_lags,
            estimate_bytes=(chunk, lag_batch) -> begin
                input_bytes = 6 * sizeof(T) * chunk * nsteps
                mu_bytes = sizeof(T) * chunk * nsteps
                lag_bytes = sizeof(Int32) * lag_batch
                partial_hist_bytes = chunk * lag_batch * n_bins * (sizeof(Int32) + 2 * sizeof(T))
                particle_partial_bytes = chunk * lag_batch * (sizeof(Int32) + sizeof(T))
                campaign_bytes = n_lags * (n_bins * (sizeof(Int64) + 2 * sizeof(Float32)) + sizeof(Int64) * 2 + sizeof(Float32) * 3)
                return input_bytes + mu_bytes + lag_bytes + partial_hist_bytes + particle_partial_bytes + campaign_bytes + Int64(64 * 1024^2)
            end,
        ) : nothing
        backend == :gpu && print_gpu_work_plan("D_mumu", memory_plan)
        backend == :gpu && (cfg[:resolved_gpu_lag_batch_size] = memory_plan.lag_batch_size)
        backend == :gpu && (cfg[:resolved_particle_chunk_size] = memory_plan.particle_chunk_size)
        backend == :gpu && (cfg[:estimated_peak_gpu_memory] = memory_plan.estimated_peak_gpu_memory)
        backend == :gpu && (cfg[:usable_gpu_memory] = memory_plan.usable_gpu_memory)
        backend == :gpu && (cfg[:memory_planner_fallback_used] = memory_plan.memory_planner_fallback_used)
        chunk_size = backend == :gpu ? min(memory_plan.particle_chunk_size, selected_particle_count) : min(Int(cfg[:particle_chunk_size]), selected_particle_count)
        chunk_size > 0 || error("particle_chunk_size must be positive.")
        nchunks = cld(selected_particle_count, chunk_size)
        lag_ranges = backend == :gpu ? gpu_lag_batches(lag_steps, memory_plan.lag_batch_size) : [1:n_lags]
        gpu_accumulators = backend == :gpu ? create_dmumu_gpu_accumulators(n_bins, n_lags) : nothing

        println("Trajectory HDF5: ", cfg[:trajectory_h5])
        println("Particle selection: ", cfg[:particle_selection])
        if cfg[:particle_selection] == :random
            println("Particle random seed: ", cfg[:particle_seed])
            println("Particles selected: ", selected_particle_count, " random indices from ", first_particle, " to ", last_particle)
        elseif cfg[:particle_selection] == :block_random
            println("Particle random seed: ", cfg[:particle_seed])
            println("Particle block size: ", cfg[:particle_block_size])
            println("Particles selected: ", selected_particle_count, " particles from random contiguous blocks, index span ", first_particle, "-", last_particle)
        else
            println("Particles selected: ", first_particle, "-", last_particle, " (", selected_particle_count, ")")
        end
        println("Saved steps: ", nsteps)
        println("Lag count: ", n_lags, " from ", first(lag_steps), " to ", last(lag_steps))
        if compute_dmumu
            println("D_mumu start mode: ", start_mode)
            println("Mu bin coordinate: ", mu_bin_coordinate_name(start_mode, bin_abs))
            println("Mu bins: ", n_bins, " from ", first(mu_edges), " to ", last(mu_edges))
        else
            println("D_mumu products: disabled")
        end
        println("Particle chunk: ", chunk_size)
        println("Chunks: ", nchunks)
        compute_dmumu && println("Backend for mu reconstruction: ", backend)
        compute_dmumu && println("D_mumu accumulation backend: ", backend)
        println("Julia threads: ", Threads.nthreads(), " active pool, max thread id ", Threads.maxthreadid())

        for chunk_id in 1:nchunks
            selection_first = (chunk_id - 1) * chunk_size + 1
            selection_last = min(selected_particle_count, selection_first + chunk_size - 1)
            chunk_indices = particle_indices[selection_first:selection_last]
            println("Chunk ", chunk_id, "/", nchunks, ": ", length(chunk_indices), " selected particles, index span ", first(chunk_indices), "-", last(chunk_indices))

            momenta = read_particle_batch(momenta_dataset, chunk_indices)

            if compute_dmumu
                positions = read_particle_batch(positions_dataset, chunk_indices)
                if backend == :gpu
                    dmu_chunk = reconstruct_mu_chunk_gpu(positions, momenta, dBx, dBy, dBz, xgrid, ygrid, zgrid, cfg, T)
                    for lag_range in lag_ranges
                        process_mu_chunk_dmumu_gpu!(dmu_chunk, lag_steps, lag_range, mu_edges, gpu_accumulators, start_mode, bin_abs, cfg)
                    end
                else
                    mu_chunk = reconstruct_mu_chunk_cpu(positions, momenta, Bx, By, Bz, xgrid, ygrid, zgrid, T)
                    process_mu_chunk_dmumu_cpu!(
                        mu_chunk,
                        lag_steps,
                        mu_edges,
                        particle_counts,
                        particle_means,
                        particle_m2s,
                        pair_sum_squares,
                        pair_counts,
                        dmumu_counts,
                        dmumu_sum_delta,
                        dmumu_sum_delta2,
                        start_mode,
                        bin_abs,
                    )
                end
            end

        end

        if backend == :gpu
            CUDA.synchronize()
            copy_dmumu_gpu_accumulators!(gpu_accumulators, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts, dmumu_counts, dmumu_sum_delta, dmumu_sum_delta2)
        end

        delta_df = nothing
        mean_delta = nothing
        mean_delta2 = nothing
        drift_norm = nothing
        dmumu_raw_norm = nothing
        dmumu_centered_norm = nothing
        dmumu_raw_per_s = nothing
        dmumu_centered_per_s = nothing
        collapsed_raw_norm = nothing
        collapsed_centered_norm = nothing
        collapsed_raw_norm_count_weighted = nothing
        collapsed_centered_norm_count_weighted = nothing
        if compute_dmumu
            delta_df = build_output_dataframe(lag_steps, t_s, t_norm, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)

            mean_delta,
            mean_delta2,
            drift_norm,
            dmumu_raw_norm,
            dmumu_centered_norm,
            dmumu_raw_per_s,
            dmumu_centered_per_s = compute_dmumu_arrays_full(dmumu_counts, dmumu_sum_delta, dmumu_sum_delta2, tau_s, tau_norm, cfg)

            collapsed_raw_norm, collapsed_raw_norm_count_weighted = average_over_tau_full(dmumu_raw_norm, dmumu_counts, cfg)
            collapsed_centered_norm, collapsed_centered_norm_count_weighted = average_over_tau_full(dmumu_centered_norm, dmumu_counts, cfg)
        end
        save_combined_full_h5(
            cfg[:output_h5],
            cfg,
            backend,
            estimated_pair_visits,
            delta_df,
            mu_edges,
            mu_centers,
            lag_steps,
            tau_s,
            tau_norm,
            first_particle,
            last_particle,
            particle_indices,
            dmumu_counts,
            dmumu_sum_delta,
            dmumu_sum_delta2,
            mean_delta,
            mean_delta2,
            drift_norm,
            dmumu_raw_norm,
            dmumu_centered_norm,
            dmumu_raw_per_s,
            dmumu_centered_per_s,
            collapsed_raw_norm,
            collapsed_centered_norm,
            collapsed_raw_norm_count_weighted,
            collapsed_centered_norm_count_weighted;
            lag_grid=lag_grid,
        )
        println("Saved combined HDF5 to ", cfg[:output_h5])

        if compute_dmumu
            plot_delta_mu2(delta_df, cfg[:output_delta_png]; use_usetex=cfg[:use_usetex])
            println("Saved delta_mu2 plot to ", cfg[:output_delta_png])

            plot_dmumu_heatmap_full(
                cfg[:output_heatmap_png],
                mu_edges,
                tau_gyroperiods,
                dmumu_centered_norm;
                use_usetex=cfg[:use_usetex],
                mu_axis_label=mu_axis_label,
                title_label=heatmap_title,
            )
            println("Saved D_mumu heatmap to ", cfg[:output_heatmap_png])

            plot_collapsed_dmumu_full(
                cfg[:output_collapsed_png],
                mu_edges,
                mu_centers,
                collapsed_raw_norm,
                collapsed_centered_norm,
                collapsed_raw_norm_count_weighted,
                collapsed_centered_norm_count_weighted;
                use_usetex=cfg[:use_usetex],
                mu_axis_label=mu_axis_label,
                title_label=collapsed_title,
            )
            println("Saved D_mumu tau-average plot to ", cfg[:output_collapsed_png])
        end
    end

    return nothing
end

function main()
    cfg = parse_cli_config(ARGS)
    run_combined_full(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
