haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

include(joinpath(@__DIR__, "compute_delta_mu2_curve.jl"))

using Random

const COMBINED_FULL_CFG = Dict{Symbol, Any}(
    :trajectory_h5 => joinpath(@__DIR__, "outputs", "campaigns", "0_5", "trajectory_cache","phase_space_10000000_GeV.h5"),
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
    :lag_mode => :uniform_samples,
    :n_lag_samples => 40,
    :max_lag_steps => nothing,
    :lag_step_stride => 1,
    :n_mu_bins => 24,
    :mu_min => -1.0,
    :mu_max => 1.0,
    :min_count_per_cell => 20,
    :max_pair_visits_without_allow => 5.0e9,
    :allow_huge => true,
    :output_dir => joinpath(@__DIR__, "outputs", "campaigns", "0_5", "10000000_GeV", "full_test"),
    :output_h5 => nothing,
    :output_delta_png => nothing,
    :output_heatmap_png => nothing,
    :output_collapsed_png => nothing,
    :use_usetex => false,
)

function print_usage()
    println("""
    Usage:
      julia compute_delta_mu2_dmumu_full.jl [options]

    This standalone postprocessor reconstructs mu(t) once per particle chunk and
    uses that same chunk to accumulate both:
      - global <(Delta mu)^2>(tau)
      - full-pair D_mumu(mu,tau) and tau-averaged D_mumu(mu)

    Important: D_mumu is exact over every selected particle and every valid
    start time for each selected lag. It does not randomly sample pairs.

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
      --n-lag-samples=N
      --max-lag-steps=N|none
      --lag-mode=uniform|stride
      --lag-step-stride=N
      --n-mu-bins=N
      --field-subset=none|NX,NY,NZ
      --allow-huge
      --smoke

    Safety:
      Large exact full-pair runs are blocked unless --allow-huge is supplied.
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
    mode = Symbol(lowercase(strip(value)))
    if mode == :uniform
        return :uniform_samples
    elseif mode in (:uniform_samples, :stride)
        return mode
    end
    error("--lag-mode must be uniform or stride")
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
            cfg[:lag_mode] = :uniform_samples
            cfg[:n_lag_samples] = 5
            cfg[:max_lag_steps] = 20
            cfg[:n_mu_bins] = 8
            cfg[:min_count_per_cell] = 2
            cfg[:field_subset] = (16, 16, 16)
            cfg[:output_dir] = joinpath(@__DIR__, "outputs", "combined_full", "smoke")
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
    max_lag = cfg[:max_lag_steps] === nothing ? nsteps - 1 : min(Int(cfg[:max_lag_steps]), nsteps - 1)
    max_lag >= 1 || error("Need at least two saved trajectory steps.")

    if cfg[:lag_mode] == :uniform_samples
        n_lags = min(Int(cfg[:n_lag_samples]), max_lag)
        return unique(round.(Int, range(1, max_lag, length=n_lags)))
    elseif cfg[:lag_mode] == :stride
        stride = Int(cfg[:lag_step_stride])
        stride >= 1 || error("lag_step_stride must be >= 1")
        return collect(1:stride:max_lag)
    end

    error("Unknown lag_mode: $(cfg[:lag_mode])")
end

function build_mu_edges_full(cfg)
    mu_minimum = Float64(cfg[:mu_min])
    mu_maximum = Float64(cfg[:mu_max])
    n_bins = Int(cfg[:n_mu_bins])
    mu_minimum < mu_maximum || error("mu_min must be less than mu_max.")
    n_bins > 0 || error("n_mu_bins must be positive.")
    edges = collect(range(mu_minimum, mu_maximum, length=n_bins + 1))
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return edges, centers
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

function estimate_pair_visits(nsteps::Integer, selected_particle_count::Integer, lag_steps)
    visits_per_particle = sum(nsteps - lag_step for lag_step in lag_steps)
    return Int128(selected_particle_count) * Int128(visits_per_particle)
end

function guard_exact_pair_cost!(nsteps::Integer, selected_particle_count::Integer, lag_steps, cfg)
    estimated_visits = estimate_pair_visits(nsteps, selected_particle_count, lag_steps)
    limit = Int128(round(Int64, Float64(cfg[:max_pair_visits_without_allow])))
    println("Estimated exact pair visits: ", format_count(estimated_visits))

    if estimated_visits > limit && !Bool(cfg[:allow_huge])
        error("""
        Refusing to start a very large exact full-pair D_mumu calculation.
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

                bin_index = mu_bin_index_full(mu_start, mu_edges)
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
        file["lag_mode"] = string(cfg[:lag_mode])
        file["n_lag_samples"] = length(lag_steps)
        file["lag_step_stride"] = Int(cfg[:lag_step_stride])
        file["max_lag_steps"] = cfg[:max_lag_steps] === nothing ? -1 : Int(cfg[:max_lag_steps])
        file["n_mu_bins"] = Int(cfg[:n_mu_bins])
        file["min_count_per_cell"] = Int(cfg[:min_count_per_cell])
        file["box_length_pc"] = Float64(cfg[:box_length_pc])
        file["field_subset"] = cfg[:field_subset] === nothing ? "nothing" : string(cfg[:field_subset])
        file["estimated_pair_visits"] = string(estimated_pair_visits)
        file["estimator_note"] = "Full-pair exact accumulation over every selected particle and every valid start time for each selected lag; no random pair sampling. mu(t) is reconstructed once per particle chunk and reused for both delta_mu2 and D_mumu."
        file["tau_average_note"] = "Lag average over the selected tau grid; this is an effective lag-averaged scattering measure, not a fitted diffusive plateau."

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
    end

    return nothing
end

function plot_dmumu_heatmap_full(path_png::AbstractString, mu_edges, tau_norm, dmumu_centered_norm; use_usetex::Bool=false)
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
    ylabel(raw"$\mu_0$")
    title(raw"Full-pair $D_{\mu\mu}(\mu,\tau)$")
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
    return nothing
end

function plot_collapsed_dmumu_full(
    path_png::AbstractString,
    mu_centers,
    collapsed_raw_norm,
    collapsed_centered_norm,
    collapsed_raw_norm_count_weighted,
    collapsed_centered_norm_count_weighted;
    use_usetex::Bool=false,
)
    mkpath(dirname(path_png))
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(7, 4))
    plot(mu_centers, collapsed_centered_norm, "o-", color="black", linewidth=1.5, markersize=4, label="centered, tau-average")
    plot(mu_centers, collapsed_raw_norm, "s--", color="tab:blue", linewidth=1.2, markersize=3, label="raw, tau-average")
    plot(mu_centers, collapsed_centered_norm_count_weighted, ":", color="0.35", linewidth=1.4, label="centered, count-weighted")
    plot(mu_centers, collapsed_raw_norm_count_weighted, "-.", color="tab:orange", linewidth=1.1, label="raw, count-weighted")
    xlabel(raw"$\mu_0$")
    ylabel(raw"$\langle D_{\mu\mu}/\Omega_0 \rangle_\tau$")
    title(raw"Full-pair tau-averaged $D_{\mu\mu}(\mu)$")
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

    backend = resolve_backend(cfg)
    T = cfg[:compute_precision]
    mu_edges, mu_centers = build_mu_edges_full(cfg)

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
        println("Mu bins: ", n_bins)
        println("Particle chunk: ", chunk_size)
        println("Chunks: ", nchunks)
        println("Backend for mu reconstruction: ", backend)
        println("Full-pair accumulation backend: cpu")
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

            process_mu_chunk_full_pairs_cpu!(
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
            )
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
        )
        println("Saved combined HDF5 to ", cfg[:output_h5])

        plot_delta_mu2(delta_df, cfg[:output_delta_png]; use_usetex=cfg[:use_usetex])
        println("Saved delta_mu2 plot to ", cfg[:output_delta_png])

        plot_dmumu_heatmap_full(cfg[:output_heatmap_png], mu_edges, tau_norm, dmumu_centered_norm; use_usetex=cfg[:use_usetex])
        println("Saved D_mumu heatmap to ", cfg[:output_heatmap_png])

        plot_collapsed_dmumu_full(
            cfg[:output_collapsed_png],
            mu_centers,
            collapsed_raw_norm,
            collapsed_centered_norm,
            collapsed_raw_norm_count_weighted,
            collapsed_centered_norm_count_weighted;
            use_usetex=cfg[:use_usetex],
        )
        println("Saved D_mumu tau-average plot to ", cfg[:output_collapsed_png])
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
