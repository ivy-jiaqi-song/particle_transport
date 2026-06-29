haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

include(joinpath(@__DIR__, "compute_delta_mu2_curve.jl"))

using Random
using Printf

const C_LIGHT = 2.99792458e8
const M_P = 1.67262192369e-27
const GEV_TO_J = 1.0e9 * 1.602176634e-19

const COMBINED_FULL_CFG = Dict{Symbol, Any}(
    :trajectory_h5 => joinpath(PIPELINE_ROOT, "outputs", "campaigns", "0_5", "trajectory_cache","phase_space_10000000_GeV.h5"),
    :turbulence_h5 => raw"/data/multiphase/MP_WeakB_0_5tcs.h5",
    :B_paths => ("i_mag_field", "j_mag_field", "k_mag_field"),
    :box_length_pc => 200.0,
    :field_subset => nothing,
    :compute_backend => :gpu,
    :compute_precision => Float32,
    :gpu_threads => 256,
    :particle_chunk_size => 128,
    :first_particle => 1,
    :n_particles_to_use => 10000,
    :particle_selection => :block_random,
    :particle_seed => 20260423,
    :particle_block_size => 128,
    :dmumu_start_mode => :sliding,
    :lag_mode => :uniform_samples,
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
    :output_dpp_png => nothing,
    :output_energy_hist_png => nothing,
    :compute_dpp => false,
    :n_energy_snapshots => 5,
    :energy_hist_bins => 60,
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
      --n-lag-samples=N
      --max-lag-steps=N|none
      --lag-mode=uniform|stride
      --lag-step-stride=N
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
        elseif startswith(argument, "--n-lag-samples=")
            cfg[:n_lag_samples] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--max-lag-steps=")
            cfg[:max_lag_steps] = parse_maybe_int(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-mode=")
            cfg[:lag_mode] = parse_lag_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-step-stride=")
            cfg[:lag_step_stride] = parse(Int, split(argument, "=", limit=2)[2])
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
    cfg[:output_dpp_png] === nothing && (cfg[:output_dpp_png] = joinpath(cfg[:output_dir], "dpp_tau_average_full.png"))
    cfg[:output_energy_hist_png] === nothing && (cfg[:output_energy_hist_png] = joinpath(cfg[:output_dir], "energy_distribution_snapshots.png"))

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

function process_momenta_chunk_dpp_cpu!(
    momenta,
    lag_steps::Vector{Int},
    p0::Float64,
    counts,
    sum_delta_p,
    sum_delta_p2,
    sum_delta_p_norm,
    sum_delta_p_norm2,
)
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

    return mean_delta_p,
           mean_delta_p2,
           mean_delta_p_norm,
           mean_delta_p_norm2,
           dpp_raw_per_s,
           dpp_centered_per_s,
           dpp_raw_norm,
           dpp_centered_norm
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
    delta_df::DataFrame,
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
    collapsed_centered_norm_count_weighted,
    dpp_results,
    energy_snapshot_results,
)
    mkpath(dirname(path_h5))
    h5open(path_h5, "w") do file
        file["trajectory_h5"] = string(cfg[:trajectory_h5])
        file["turbulence_h5"] = string(cfg[:turbulence_h5])
        file["compute_backend"] = string(backend)
        file["compute_precision"] = string(cfg[:compute_precision])
        file["particle_chunk_size"] = Int(cfg[:particle_chunk_size])
        file["first_particle"] = Int(first_particle)
        file["last_particle"] = Int(last_particle)
        file["n_particles_used"] = length(particle_indices)
        file["particle_selection"] = string(cfg[:particle_selection])
        file["particle_seed"] = Int(cfg[:particle_seed])
        file["particle_block_size"] = Int(cfg[:particle_block_size])
        file["particle_indices"] = particle_indices
        start_mode = dmumu_start_mode(cfg)
        bin_abs = mu_bin_abs(cfg)
        file["dmumu_start_mode"] = string(start_mode)
        file["mu_bin_abs"] = bin_abs
        file["mu_bin_coordinate"] = mu_bin_coordinate_name(start_mode, bin_abs)
        file["lag_mode"] = string(cfg[:lag_mode])
        file["n_lag_samples"] = length(lag_steps)
        file["requested_n_lag_samples"] = Int(cfg[:n_lag_samples])
        file["min_lag_steps"] = Int(get(cfg, :min_lag_steps, 1))
        file["lag_step_stride"] = Int(cfg[:lag_step_stride])
        file["max_lag_steps"] = cfg[:max_lag_steps] === nothing ? -1 : Int(cfg[:max_lag_steps])
        file["n_mu_bins"] = Int(cfg[:n_mu_bins])
        file["min_count_per_cell"] = Int(cfg[:min_count_per_cell])
        file["box_length_pc"] = Float64(cfg[:box_length_pc])
        file["field_subset"] = cfg[:field_subset] === nothing ? "nothing" : string(cfg[:field_subset])
        file["estimated_pair_visits"] = string(estimated_pair_visits)
        start_note = start_mode == :injection ?
            "Injection-anchored accumulation: each selected particle contributes at most one pair per selected lag, Delta mu = mu(tau) - mu(0), binned by injected mu(0)." :
            "Sliding full-pair accumulation over every selected particle and every valid start time for each selected lag; Delta mu = mu(t + tau) - mu(t), binned by pair-start mu(t)."
        bin_note = bin_abs ?
            " D_mumu bins use the absolute value of the start mu; stored mu values and Delta mu remain signed." :
            " D_mumu bins use the signed start mu."
        file["estimator_note"] = start_note * bin_note * " No random pair sampling. mu(t) is reconstructed once per particle chunk and reused for both delta_mu2 and D_mumu."
        file["tau_average_note"] = "Lag average over the selected tau grid; this is an effective lag-averaged scattering measure, not a fitted diffusive plateau."
        file["dpp_note"] = "Global scalar momentum diffusion uses p = sqrt(px^2 + py^2 + pz^2) and Delta p = p(t + tau) - p(t), accumulated over the same selected particles and sliding start times as D_mumu. Normalized D_pp uses Delta p / p0 and tau * Omega0."

        delta_group = create_group(file, "delta_mu2")
        for column_name in names(delta_df)
            delta_group[string(column_name)] = collect(delta_df[!, column_name])
        end

        dmumu_group = create_group(file, "dmumu")
        dmumu_group["mu_edges"] = mu_edges
        dmumu_group["mu_centers"] = mu_centers
        dmumu_group["lag_step"] = lag_steps
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

        if dpp_results !== nothing
            dpp_group = create_group(file, "dpp")
            dpp_group["lag_step"] = lag_steps
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
        end

        if energy_snapshot_results !== nothing
            energy_group = create_group(file, "energy_snapshots")
            energy_group["snapshot_step_index"] = energy_snapshot_results.indices
            energy_group["snapshot_t_s"] = energy_snapshot_results.t_s
            energy_group["snapshot_t_norm"] = energy_snapshot_results.t_norm
            energy_group["energy_GeV"] = energy_snapshot_results.energy_gev
            energy_group["particle_indices"] = particle_indices
        end
    end

    return nothing
end

function plot_dpp_tau_full(path_png::AbstractString, tau_norm, dpp_raw_norm, dpp_centered_norm; use_usetex::Bool=false)
    mkpath(dirname(path_png))
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(7, 4))
    plot(tau_norm, dpp_centered_norm, "o-", color="black", linewidth=1.5, markersize=4, label="centered")
    plot(tau_norm, dpp_raw_norm, "s--", color="tab:blue", linewidth=1.2, markersize=3, label="raw")
    xlabel(raw"$\tau\Omega_0$")
    ylabel(raw"$D_{pp}/(p_0^2\Omega_0)$")
    title(raw"Global $D_{pp}(\tau)$")
    grid(true, alpha=0.3)
    legend(frameon=false, fontsize=8)
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
    return nothing
end

function plot_energy_histograms(path_png::AbstractString, snapshot_t_norm, energy_gev; bins::Integer=60, use_usetex::Bool=false)
    mkpath(dirname(path_png))
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(7.5, 4.5))
    snapshot_count = size(energy_gev, 1)
    for snapshot_index in 1:snapshot_count
        values = Float64[value for value in energy_gev[snapshot_index, :] if isfinite(value)]
        isempty(values) && continue
        hist(values, bins=Int(bins), histtype="step", linewidth=1.4, density=true, label=@sprintf("tOmega0 = %.3g", snapshot_t_norm[snapshot_index]))
    end
    xlabel("Kinetic energy [GeV]")
    ylabel("Probability density")
    title("Particle energy distribution snapshots")
    grid(true, alpha=0.25)
    legend(frameon=false, fontsize=8)
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
    return nothing
end

function plot_dmumu_heatmap_full(
    path_png::AbstractString,
    mu_edges,
    tau_norm,
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
        extent=(minimum(tau_norm), maximum(tau_norm), mu_edges[1], mu_edges[end]),
        cmap="viridis",
    )
    colorbar(label=raw"$D_{\mu\mu}/\Omega_0$ (centered)")
    xlabel(raw"$\tau\Omega_0$")
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
    mkpath(dirname(cfg[:output_delta_png]))
    mkpath(dirname(cfg[:output_heatmap_png]))
    mkpath(dirname(cfg[:output_collapsed_png]))
    Bool(get(cfg, :compute_dpp, false)) && mkpath(dirname(cfg[:output_dpp_png]))
    Bool(get(cfg, :compute_dpp, false)) && mkpath(dirname(cfg[:output_energy_hist_png]))

    backend = resolve_backend(cfg)
    T = cfg[:compute_precision]
    mu_edges, mu_centers = build_mu_edges_full(cfg)
    start_mode = dmumu_start_mode(cfg)
    bin_abs = mu_bin_abs(cfg)
    mu_axis_label = mu_bin_axis_label(start_mode, bin_abs)
    heatmap_title = dmumu_plot_title(bin_abs)
    collapsed_title = dmumu_tau_average_title(bin_abs)

    println("Loading magnetic field from ", cfg[:turbulence_h5])
    Bx, By, Bz = load_B_fields(cfg, T)
    nx, ny, nz = size(Bx)
    xgrid, ygrid, zgrid = build_uniform_coords(cfg, nx, ny, nz, T)

    dBx = nothing
    dBy = nothing
    dBz = nothing
    if backend == :gpu
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
        nsteps = validate_trajectory_layout(positions_dataset, momenta_dataset, t_s, t_norm)
        total_particles = size(positions_dataset, 1)

        particle_indices = build_particle_indices(total_particles, cfg)
        first_particle = first(particle_indices)
        last_particle = last(particle_indices)
        selected_particle_count = length(particle_indices)
        selected_particle_count > 0 || error("No particles selected.")

        lag_steps = build_selected_lag_steps(nsteps, cfg)
        tau_s = [Float64(t_s[lag_step + 1] - t_s[1]) for lag_step in lag_steps]
        tau_norm = [Float64(t_norm[lag_step + 1] - t_norm[1]) for lag_step in lag_steps]
        estimated_pair_visits = guard_exact_pair_cost!(nsteps, selected_particle_count, lag_steps, cfg)

        n_lags = length(lag_steps)
        n_bins = length(mu_centers)
        particle_counts = zeros(Int, n_lags)
        particle_means = zeros(Float64, n_lags)
        particle_m2s = zeros(Float64, n_lags)
        pair_sum_squares = zeros(Float64, n_lags)
        pair_counts = zeros(Int64, n_lags)
        dmumu_counts = zeros(Int64, n_bins, n_lags)
        dmumu_sum_delta = zeros(Float64, n_bins, n_lags)
        dmumu_sum_delta2 = zeros(Float64, n_bins, n_lags)
        compute_dpp = Bool(get(cfg, :compute_dpp, false))
        p0 = compute_dpp ? momentum_magnitude(momenta_dataset[particle_indices[1], 1, 1], momenta_dataset[particle_indices[1], 2, 1], momenta_dataset[particle_indices[1], 3, 1]) : NaN
        compute_dpp && !(isfinite(p0) && p0 > 0.0) && error("Cannot compute D_pp normalization: first selected particle has invalid initial momentum.")
        dpp_counts = zeros(Int64, n_lags)
        dpp_sum_delta_p = zeros(Float64, n_lags)
        dpp_sum_delta_p2 = zeros(Float64, n_lags)
        dpp_sum_delta_p_norm = zeros(Float64, n_lags)
        dpp_sum_delta_p_norm2 = zeros(Float64, n_lags)
        snapshot_indices = compute_dpp ? selected_energy_snapshot_indices(nsteps, Int(get(cfg, :n_energy_snapshots, 5))) : Int[]
        energy_snapshots = compute_dpp ? fill(NaN, length(snapshot_indices), selected_particle_count) : Array{Float64}(undef, 0, 0)

        chunk_size = min(Int(cfg[:particle_chunk_size]), selected_particle_count)
        chunk_size > 0 || error("particle_chunk_size must be positive.")
        nchunks = cld(selected_particle_count, chunk_size)

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
        println("D_mumu start mode: ", start_mode)
        println("Mu bin coordinate: ", mu_bin_coordinate_name(start_mode, bin_abs))
        println("Mu bins: ", n_bins, " from ", first(mu_edges), " to ", last(mu_edges))
        println("Particle chunk: ", chunk_size)
        println("Chunks: ", nchunks)
        println("Backend for mu reconstruction: ", backend)
        println("D_mumu accumulation backend: cpu")
        compute_dpp && println("D_pp and energy snapshots: enabled with ", length(snapshot_indices), " snapshots")
        println("Julia threads: ", Threads.nthreads(), " active pool, max thread id ", Threads.maxthreadid())

        for chunk_id in 1:nchunks
            selection_first = (chunk_id - 1) * chunk_size + 1
            selection_last = min(selected_particle_count, selection_first + chunk_size - 1)
            chunk_indices = particle_indices[selection_first:selection_last]
            println("Chunk ", chunk_id, "/", nchunks, ": ", length(chunk_indices), " selected particles, index span ", first(chunk_indices), "-", last(chunk_indices))

            positions = read_particle_batch(positions_dataset, chunk_indices)
            momenta = read_particle_batch(momenta_dataset, chunk_indices)

            if backend == :gpu
                dmu_chunk = reconstruct_mu_chunk_gpu(positions, momenta, dBx, dBy, dBz, xgrid, ygrid, zgrid, cfg, T)
                mu_chunk = Array(dmu_chunk)
                GC.gc(false)
                CUDA.reclaim()
            else
                mu_chunk = reconstruct_mu_chunk_cpu(positions, momenta, Bx, By, Bz, xgrid, ygrid, zgrid, T)
            end

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

            if compute_dpp
                process_momenta_chunk_dpp_cpu!(
                    momenta,
                    lag_steps,
                    p0,
                    dpp_counts,
                    dpp_sum_delta_p,
                    dpp_sum_delta_p2,
                    dpp_sum_delta_p_norm,
                    dpp_sum_delta_p_norm2,
                )
                fill_energy_snapshots!(energy_snapshots, momenta, snapshot_indices, selection_first)
            end
        end

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
        dpp_results = nothing
        energy_snapshot_results = nothing
        if compute_dpp
            mean_delta_p,
            mean_delta_p2,
            mean_delta_p_norm,
            mean_delta_p_norm2,
            dpp_raw_per_s,
            dpp_centered_per_s,
            dpp_raw_norm,
            dpp_centered_norm = compute_dpp_arrays(
                dpp_counts,
                dpp_sum_delta_p,
                dpp_sum_delta_p2,
                dpp_sum_delta_p_norm,
                dpp_sum_delta_p_norm2,
                tau_s,
                tau_norm,
            )
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
                energy_gev = energy_snapshots,
            )
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
            collapsed_centered_norm_count_weighted,
            dpp_results,
            energy_snapshot_results,
        )
        println("Saved combined HDF5 to ", cfg[:output_h5])

        plot_delta_mu2(delta_df, cfg[:output_delta_png]; use_usetex=cfg[:use_usetex])
        println("Saved delta_mu2 plot to ", cfg[:output_delta_png])

        plot_dmumu_heatmap_full(
            cfg[:output_heatmap_png],
            mu_edges,
            tau_norm,
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

        if compute_dpp
            plot_dpp_tau_full(cfg[:output_dpp_png], tau_norm, dpp_results.dpp_raw_norm, dpp_results.dpp_centered_norm; use_usetex=cfg[:use_usetex])
            println("Saved D_pp tau plot to ", cfg[:output_dpp_png])
            plot_energy_histograms(cfg[:output_energy_hist_png], energy_snapshot_results.t_norm, energy_snapshot_results.energy_gev; bins=Int(get(cfg, :energy_hist_bins, 60)), use_usetex=cfg[:use_usetex])
            println("Saved energy histogram plot to ", cfg[:output_energy_hist_png])
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
