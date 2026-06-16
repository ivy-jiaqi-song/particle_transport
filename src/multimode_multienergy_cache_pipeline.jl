haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

const PIPELINE_ROOT = dirname(@__DIR__)

module RunnerMod
include(joinpath(@__DIR__, "phase_space_gpu_runner.jl"))
end

module CombinedFullMod
include(joinpath(@__DIR__, "compute_delta_mu2_dmumu_full.jl"))
end

using CUDA
using HDF5
using Printf
using Statistics
using TOML

const MODE_DECOMPOSITION_ROOT = raw"/home/user0001/MHDFlows_replicate/multiphase_mode_decomposition/outputs"
const TURBULENCE_TAGS = ("0_5", "0_9", "1_0")
const MODE_NAMES = ("alfven", "fast", "slow")

const MHD512_DATASET_TAG = "512_a_00100"
const MHD512_TOTAL_H5 = raw"/home/user0001/MHDFlows_replicate/h5_outputs/512_a.00100.h5"
const MHD512_MODE_DIR = raw"/home/user0001/MHDFlows_replicate/mhdflows512_mode_decomposition/outputs/512_a_00100_cs10_L200/mode_h5"

const CACHE_PIPELINE_CFG = Dict{Symbol, Any}(
    :cache_mode => :mu,
    :compute_dmumu => true,
    :mode_campaigns => nothing,
    :energies => [1e5, 1e6, 1e7],
    :delete_cache_on_success => true,
    :reuse_existing_cache => true,
    :skip_completed_outputs => true,
    :stop_on_error => true,
    :run_all_particles_dmumu => false,
    :mu_cache_output_precision => Float32,
    :trajectory_overrides => Dict{Symbol, Any}(
        :B_paths => ("i_mag_field", "j_mag_field", "k_mag_field"),
        :v_paths => ("i_velocity", "j_velocity", "k_velocity"),
        :velocity_unit_in_m_per_s => 1e3,
        :box_length_pc => 200.0,
        :eta => 0.1,
        :tOmega0_max => 5000.0,
        :use_cfl => false,
        :cfl => 0.2,
        :seed => 42,
        :n_particles => 100000,
        :precision => Float64,
        :field_subset => nothing,
        :boundary => :periodic,
        :trajectory_time_stride => 1,
        :trajectory_output_precision => Float32,
        :progress_every => 5000,
    ),
    :combined_overrides => Dict{Symbol, Any}(
        :B_paths => ("i_mag_field", "j_mag_field", "k_mag_field"),
        :box_length_pc => 200.0,
        :field_subset => nothing,
        :compute_backend => :auto,
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
        :use_usetex => false,
    ),
)

function default_input_spec()
    return Dict{Symbol, Any}(
        :h5_dir => nothing,
        :label => nothing,
        :medium => "iso",
        :file_stem => nothing,
        :total_file => nothing,
        :mode_file_pattern => nothing,
    )
end

function default_input_paths()
    return Dict{Symbol, Any}(
        :mode_decomposition_root => MODE_DECOMPOSITION_ROOT,
        :mhd512_total_h5 => MHD512_TOTAL_H5,
        :mhd512_mode_dir => MHD512_MODE_DIR,
    )
end

function shallow_copy_dict(dict_like)
    return Dict{Symbol, Any}(pair.first => pair.second for pair in pairs(dict_like))
end

function copy_campaigns(campaigns)
    return [shallow_copy_dict(campaign) for campaign in campaigns]
end

function merge_cfg(base_cfg, overrides)
    merged = shallow_copy_dict(base_cfg)
    merge!(merged, overrides)
    return merged
end

function energy_tag(energy_GeV)
    return string(Int(round(Float64(energy_GeV)))) * "_GeV"
end

function cache_mode_label(cache_mode::Symbol)
    cache_mode == :mu && return "mu"
    cache_mode == :phase_space && return "phase_space"
    error("Unknown cache mode: " * string(cache_mode))
end

function parse_cache_mode(value::AbstractString)
    normalized = replace(lowercase(strip(value)), "-" => "_")
    normalized == "mu" && return :mu
    normalized in ("phase", "phase_space") && return :phase_space
    error("--cache-mode must be mu or phase-space")
end

function parse_input_layout(value::AbstractString)
    normalized = replace(lowercase(strip(value)), "-" => "_")
    normalized in ("mp_weakb", "weakb", "multiphase") && return :mp_weakb
    normalized in ("mhd512", "512", "512_a_00100") && return :mhd512
    error("--layout must be mp-weakb or mhd512")
end

function normalize_medium(value)
    normalized = replace(lowercase(strip(String(value))), "-" => "_")
    normalized in ("iso", "isothermal") && return "iso"
    normalized in ("mp", "multiphase", "multi_phase") && return "mp"
    error("[input].medium must be iso/isothermal or mp/multiphase.")
end

function split_csv_selector(value::AbstractString; flag_name::AbstractString)
    cleaned = strip(value)
    lowercase(cleaned) == "all" && return nothing

    values = String[]
    for part in split(cleaned, ",")
        item = strip(part)
        isempty(item) && continue
        push!(values, item)
    end
    isempty(values) && error(flag_name * " requires at least one non-empty value or 'all'.")
    return values
end

function normalize_campaign_selector(value::AbstractString)
    normalized = strip(value)
    normalized = replace(normalized, ":" => "/")
    return normalized
end

function parse_campaign_selector(value::AbstractString)
    parsed = split_csv_selector(value; flag_name="--campaign")
    parsed === nothing && return nothing
    return normalize_campaign_selector.(parsed)
end

function mode_h5_path(input_paths, turbulence_tag::AbstractString, mode_name::AbstractString)
    full_tag = "MP_WeakB_" * turbulence_tag * "tcs"
    return joinpath(
        input_paths[:mode_decomposition_root],
        full_tag * "_cs1_L200",
        "mode_h5",
        full_tag * "_" * mode_name * ".h5",
    )
end

function input_file_path(input_spec, filename)
    filename = String(filename)
    return isabspath(filename) ? normpath(filename) : normpath(joinpath(input_spec[:h5_dir], filename))
end

function input_label(input_spec)
    label = input_spec[:label]
    label === nothing && error("[input].label is required for generic input configuration.")
    label = strip(String(label))
    isempty(label) && error("[input].label cannot be empty.")
    return label
end

function input_file_stem(input_spec)
    stem = input_spec[:file_stem]
    stem === nothing && error("[input].file_stem is required unless explicit file names are provided.")
    stem = strip(String(stem))
    isempty(stem) && error("[input].file_stem cannot be empty.")
    return stem
end

function format_input_pattern(pattern, stem::AbstractString, mode_name::AbstractString)
    filename = replace(String(pattern), "{stem}" => stem)
    filename = replace(filename, "{mode}" => mode_name)
    return filename
end

function generic_total_h5_path(input_spec)
    if input_spec[:total_file] !== nothing
        return input_file_path(input_spec, input_spec[:total_file])
    end
    return input_file_path(input_spec, input_file_stem(input_spec) * ".h5")
end

function generic_mode_h5_path(input_spec, mode_name::AbstractString)
    stem = input_file_stem(input_spec)
    filename = input_spec[:mode_file_pattern] === nothing ?
        stem * "_" * mode_name * ".h5" :
        format_input_pattern(input_spec[:mode_file_pattern], stem, mode_name)
    return input_file_path(input_spec, filename)
end

function build_generic_campaigns(root_prefix::AbstractString, input_spec, mode_decomposition_available::Bool, available_modes)
    input_spec[:h5_dir] === nothing && error("[input].h5_dir is required for generic input configuration.")
    isdir(input_spec[:h5_dir]) || error("[input].h5_dir does not exist: " * String(input_spec[:h5_dir]))

    label = input_label(input_spec)
    medium = normalize_medium(input_spec[:medium])
    base_root = joinpath(root_prefix, medium, label)

    campaigns = Dict{Symbol, Any}[]
    if mode_decomposition_available
        for mode_name in available_modes
            mode_name = strip(String(mode_name))
            isempty(mode_name) && error("[run].available_modes cannot contain empty mode names.")
            push!(
                campaigns,
                Dict{Symbol, Any}(
                    :campaign_tag => medium * "/" * label * "/" * mode_name,
                    :campaign_aliases => [label * "/" * mode_name],
                    :medium => medium,
                    :turbulence_tag => label,
                    :mode_name => mode_name,
                    :campaign_root => joinpath(base_root, mode_name),
                    :turbulence_h5 => generic_mode_h5_path(input_spec, mode_name),
                ),
            )
        end
    else
        push!(
            campaigns,
            Dict{Symbol, Any}(
                :campaign_tag => medium * "/" * label * "/total",
                :campaign_aliases => [label * "/total", label],
                :medium => medium,
                :turbulence_tag => label,
                :mode_name => "total",
                :campaign_root => joinpath(base_root, "total"),
                :turbulence_h5 => generic_total_h5_path(input_spec),
            ),
        )
    end
    return campaigns
end

function build_mode_campaigns(root_prefix::AbstractString, input_paths)
    campaigns = Dict{Symbol, Any}[]
    for turbulence_tag in TURBULENCE_TAGS
        for mode_name in MODE_NAMES
            push!(
                campaigns,
                Dict{Symbol, Any}(
                    :campaign_tag => turbulence_tag * "/" * mode_name,
                    :turbulence_tag => turbulence_tag,
                    :mode_name => mode_name,
                    :campaign_root => joinpath(root_prefix, turbulence_tag, mode_name),
                    :turbulence_h5 => mode_h5_path(input_paths, turbulence_tag, mode_name),
                ),
            )
        end
    end
    return campaigns
end

function mhd512_mode_h5_path(input_paths, mode_name::AbstractString)
    return joinpath(input_paths[:mhd512_mode_dir], MHD512_DATASET_TAG * "_" * mode_name * ".h5")
end

function build_mhd512_campaigns(root_prefix::AbstractString, input_paths)
    campaigns = Dict{Symbol, Any}[]
    push!(
        campaigns,
        Dict{Symbol, Any}(
            :campaign_tag => MHD512_DATASET_TAG * "/total",
            :turbulence_tag => MHD512_DATASET_TAG,
            :mode_name => "total",
            :campaign_root => joinpath(root_prefix, MHD512_DATASET_TAG, "total"),
            :turbulence_h5 => input_paths[:mhd512_total_h5],
        ),
    )

    for mode_name in MODE_NAMES
        push!(
            campaigns,
            Dict{Symbol, Any}(
                :campaign_tag => MHD512_DATASET_TAG * "/" * mode_name,
                :turbulence_tag => MHD512_DATASET_TAG,
                :mode_name => mode_name,
                :campaign_root => joinpath(root_prefix, MHD512_DATASET_TAG, mode_name),
                :turbulence_h5 => mhd512_mode_h5_path(input_paths, mode_name),
            ),
        )
    end
    return campaigns
end

function build_legacy_campaigns(input_layout::Symbol, root_prefix::AbstractString, input_paths)
    input_layout == :mp_weakb && return build_mode_campaigns(root_prefix, input_paths)
    input_layout == :mhd512 && return build_mhd512_campaigns(root_prefix, input_paths)
    error("Unknown input layout: " * string(input_layout))
end

function build_campaigns(cfg, root_prefix::AbstractString)
    if cfg[:input_kind] == :generic
        return build_generic_campaigns(
            root_prefix,
            cfg[:input_spec],
            Bool(cfg[:mode_decomposition_available]),
            cfg[:available_modes],
        )
    end
    return build_legacy_campaigns(cfg[:input_layout], root_prefix, cfg[:input_paths])
end

function campaign_selector_values(campaign)
    values = String[campaign[:campaign_tag]]
    if haskey(campaign, :campaign_aliases)
        append!(values, String.(campaign[:campaign_aliases]))
    end
    return values
end

function validate_selected_values(campaigns, selected_values, key::Symbol, flag_name::AbstractString)
    selected_values === nothing && return nothing
    available = key == :campaign_tag ? unique(vcat([campaign_selector_values(campaign) for campaign in campaigns]...)) : unique([campaign[key] for campaign in campaigns])
    missing = [value for value in selected_values if !(value in available)]
    isempty(missing) && return nothing
    error("Unknown " * flag_name * " value(s): " * join(missing, ", ") * ". Available: " * join(available, ", "))
end

function campaign_matches_selector(campaign, selector::AbstractString)
    return selector in campaign_selector_values(campaign)
end

function filter_campaigns(campaigns, selected_turbulences, selected_modes, selected_campaigns)
    filtered = campaigns

    if selected_turbulences !== nothing
        wanted = Set(selected_turbulences)
        filtered = [campaign for campaign in filtered if campaign[:turbulence_tag] in wanted]
    end

    if selected_modes !== nothing
        wanted = Set(selected_modes)
        filtered = [campaign for campaign in filtered if campaign[:mode_name] in wanted]
    end

    if selected_campaigns !== nothing
        wanted = Set(selected_campaigns)
        filtered = [campaign for campaign in filtered if any(selector -> campaign_matches_selector(campaign, selector), wanted)]
    end

    isempty(filtered) && error("No campaigns selected. Check --turbulence, --mode, and --campaign.")
    return filtered
end

function filter_campaigns_by_requests(campaigns, campaign_requests, default_energies)
    campaign_requests === nothing && return nothing

    selected = Dict{Symbol, Any}[]
    for request in campaign_requests
        matches = campaigns
        if haskey(request, :campaign_tag)
            selector = request[:campaign_tag]
            matches = [campaign for campaign in matches if campaign_matches_selector(campaign, selector)]
        end
        if haskey(request, :mode_name)
            mode_name = request[:mode_name]
            matches = [campaign for campaign in matches if campaign[:mode_name] == mode_name]
        end
        isempty(matches) && error("No campaign matched configured run.campaigns entry: " * string(request))

        for campaign in matches
            copied = shallow_copy_dict(campaign)
            copied[:energies] = get(request, :energies, default_energies)
            push!(selected, copied)
        end
    end
    return selected
end

function validate_mode_files(campaigns)
    missing = [campaign[:turbulence_h5] for campaign in campaigns if !isfile(campaign[:turbulence_h5])]
    isempty(missing) && return nothing
    error("Missing input HDF5 file(s):\n" * join(missing, "\n"))
end

function ensure_parent(path::AbstractString)
    mkpath(dirname(path))
    return path
end

function verify_file_nonempty(path::AbstractString)
    isfile(path) || error("Missing file: " * path)
    filesize(path) > 0 || error("Empty file: " * path)
    return true
end

function verify_phase_space_cache_h5(path_h5::AbstractString)
    verify_file_nonempty(path_h5)
    h5open(path_h5, "r") do file
        positions = file["positions"]
        momenta = file["momenta"]
        t_s = read(file["t_s"])
        t_norm = read(file["t_norm"])
        alive_fraction = read(file["alive_fraction"])
        size(positions) == size(momenta) || error("Phase-space verification failed: positions/momenta shapes differ for " * path_h5)
        nsteps = size(positions, 3)
        length(t_s) == nsteps || error("Phase-space verification failed: t_s length mismatch for " * path_h5)
        length(t_norm) == nsteps || error("Phase-space verification failed: t_norm length mismatch for " * path_h5)
        length(alive_fraction) == nsteps || error("Phase-space verification failed: alive_fraction length mismatch for " * path_h5)
        nsteps > 1 || error("Phase-space verification failed: not enough saved steps in " * path_h5)
    end
    return true
end

function verify_mu_cache_h5(path_h5::AbstractString)
    verify_file_nonempty(path_h5)
    h5open(path_h5, "r") do file
        mu = file["mu"]
        t_s = read(file["t_s"])
        t_norm = read(file["t_norm"])
        alive_fraction = read(file["alive_fraction"])
        nsteps = size(mu, 2)
        length(t_s) == nsteps || error("Mu-cache verification failed: t_s length mismatch for " * path_h5)
        length(t_norm) == nsteps || error("Mu-cache verification failed: t_norm length mismatch for " * path_h5)
        length(alive_fraction) == nsteps || error("Mu-cache verification failed: alive_fraction length mismatch for " * path_h5)
        size(mu, 1) > 0 || error("Mu-cache verification failed: zero particles in " * path_h5)
        nsteps > 1 || error("Mu-cache verification failed: not enough saved steps in " * path_h5)
    end
    return true
end

function verify_cache_h5(path_h5::AbstractString, cache_mode::Symbol)
    cache_mode == :phase_space && return verify_phase_space_cache_h5(path_h5)
    cache_mode == :mu && return verify_mu_cache_h5(path_h5)
    error("Unknown cache mode: " * string(cache_mode))
end

function verify_combined_outputs(path_h5::AbstractString, delta_png::AbstractString, heatmap_png::AbstractString, collapsed_png::AbstractString)
    verify_file_nonempty(path_h5)
    verify_file_nonempty(delta_png)
    verify_file_nonempty(heatmap_png)
    verify_file_nonempty(collapsed_png)

    h5open(path_h5, "r") do file
        delta_group = file["delta_mu2"]
        dmumu_group = file["dmumu"]

        delta_tau_norm = read(delta_group["tau_norm"])
        mean_curve = read(delta_group["delta_mu2_particle_mean"])
        length(delta_tau_norm) == length(mean_curve) || error("Combined verification failed: delta_mu2 tau/curve mismatch for " * path_h5)
        length(delta_tau_norm) > 0 || error("Combined verification failed: no delta_mu2 lag points in " * path_h5)
        any(isfinite, mean_curve) || error("Combined verification failed: no finite delta_mu2 values in " * path_h5)

        mu_centers = read(dmumu_group["mu_centers"])
        dmumu_tau_norm = read(dmumu_group["tau_norm"])
        heatmap = read(dmumu_group["D_mumu_centered_norm"])
        collapsed = read(dmumu_group["D_mumu_centered_tau_average_norm"])
        size(heatmap, 1) == length(mu_centers) || error("Combined verification failed: dmumu mu axis mismatch for " * path_h5)
        size(heatmap, 2) == length(dmumu_tau_norm) || error("Combined verification failed: dmumu tau axis mismatch for " * path_h5)
        length(collapsed) == length(mu_centers) || error("Combined verification failed: dmumu collapsed curve mismatch for " * path_h5)
        any(isfinite, heatmap) || error("Combined verification failed: no finite dmumu heatmap values in " * path_h5)
        any(isfinite, collapsed) || error("Combined verification failed: no finite dmumu collapsed values in " * path_h5)
    end

    return true
end

function h5_scalar_value_or(file, dataset_name::AbstractString, default)
    haskey(file, dataset_name) || return default
    value = read(file[dataset_name])
    if value isa AbstractArray
        isempty(value) && return default
        return first(value)
    end
    return value
end

function h5_scalar_string_or(file, dataset_name::AbstractString, default::AbstractString)
    value = h5_scalar_value_or(file, dataset_name, default)
    return string(value)
end

function h5_scalar_int_or(file, dataset_name::AbstractString, default::Integer)
    value = h5_scalar_value_or(file, dataset_name, default)
    value isa Integer && return Int(value)
    return parse(Int, string(value))
end

function h5_scalar_bool_or(file, dataset_name::AbstractString, default::Bool)
    value = h5_scalar_value_or(file, dataset_name, default)
    value isa Bool && return value
    value isa Integer && return value != 0
    normalized = lowercase(strip(string(value)))
    normalized in ("true", "yes", "1") && return true
    normalized in ("false", "no", "0") && return false
    return default
end

function combined_outputs_match_requested(path_h5::AbstractString, cfg)
    h5open(path_h5, "r") do file
        stored_start_mode = CombinedFullMod.parse_dmumu_start_mode(h5_scalar_string_or(file, "dmumu_start_mode", "sliding"))
        stored_mu_bin_abs = h5_scalar_bool_or(file, "mu_bin_abs", false)
        stored_lag_mode = CombinedFullMod.parse_lag_mode(h5_scalar_string_or(file, "lag_mode", "uniform_samples"))
        stored_lag_sample_count = h5_scalar_int_or(file, "n_lag_samples", Int(cfg[:n_lag_samples]))
        stored_requested_n_lag_samples = h5_scalar_int_or(file, "requested_n_lag_samples", stored_lag_sample_count)
        stored_min_lag_steps = h5_scalar_int_or(file, "min_lag_steps", 1)
        stored_lag_step_stride = h5_scalar_int_or(file, "lag_step_stride", 1)
        stored_max_lag_steps = h5_scalar_int_or(file, "max_lag_steps", -1)
        stored_n_mu_bins = h5_scalar_int_or(file, "n_mu_bins", Int(cfg[:n_mu_bins]))
        stored_mu_edges = read(file["dmumu"]["mu_edges"])

        requested_max_lag_steps = cfg[:max_lag_steps] === nothing ? -1 : Int(cfg[:max_lag_steps])
        return stored_start_mode == CombinedFullMod.dmumu_start_mode(cfg) &&
               stored_mu_bin_abs == CombinedFullMod.mu_bin_abs(cfg) &&
               stored_lag_mode == cfg[:lag_mode] &&
               stored_requested_n_lag_samples == Int(cfg[:n_lag_samples]) &&
               stored_min_lag_steps == Int(get(cfg, :min_lag_steps, 1)) &&
               stored_lag_step_stride == Int(cfg[:lag_step_stride]) &&
               stored_max_lag_steps == requested_max_lag_steps &&
               stored_n_mu_bins == Int(cfg[:n_mu_bins]) &&
               length(stored_mu_edges) == Int(cfg[:n_mu_bins]) + 1 &&
               isapprox(first(stored_mu_edges), Float64(cfg[:mu_min]); atol=0.0, rtol=0.0) &&
               isapprox(last(stored_mu_edges), Float64(cfg[:mu_max]); atol=0.0, rtol=0.0)
    end
end

function compact_outputs_complete(paths, cfg)
    try
        verify_combined_outputs(paths.combined_h5, paths.delta_png, paths.dmumu_heatmap_png, paths.dmumu_collapsed_png)
        combined_outputs_match_requested(paths.combined_h5, cfg) || return false
        return true
    catch
        return false
    end
end

function delete_cache_if_requested(path_h5::AbstractString, cfg)
    if cfg[:delete_cache_on_success]
        rm(path_h5; force=false)
        println("Deleted cache file ", path_h5)
        return true
    end
    println("Keeping cache file ", path_h5)
    return false
end

function summary_row(energy_GeV, status::AbstractString, deleted::Bool, cache_path::AbstractString, combined_h5::AbstractString, note::AbstractString)
    return (
        energy_GeV = Float64(energy_GeV),
        status = status,
        deleted = deleted,
        cache_path = cache_path,
        combined_h5 = combined_h5,
        note = note,
    )
end

function write_summary(path::AbstractString, rows)
    ensure_parent(path)
    open(path, "w") do io
        println(io, "energy_GeV\tstatus\tcache_deleted\tcache_path\tcombined_h5\tnote")
        for row in rows
            println(
                io,
                join((
                    @sprintf("%.6f", row.energy_GeV),
                    row.status,
                    row.deleted ? "true" : "false",
                    row.cache_path,
                    row.combined_h5,
                    replace(row.note, '\n' => ' '),
                ), '\t'),
            )
        end
    end
    return nothing
end

function build_energy_paths(cfg, energy_GeV)
    tag = energy_tag(energy_GeV)
    campaign_root = cfg[:campaign_root]
    cache_dir = joinpath(campaign_root, "cache")
    science_dir = joinpath(campaign_root, tag)
    cache_file = cfg[:cache_mode] == :mu ? "mu_cache_" * tag * ".h5" : "phase_space_" * tag * ".h5"
    cache_h5 = joinpath(cache_dir, cache_file)
    combined_h5 = joinpath(science_dir, "delta_mu2_dmumu_full.h5")
    delta_png = joinpath(science_dir, "delta_mu2_curve_full.png")
    dmumu_heatmap_png = joinpath(science_dir, "dmumu_mu_tau_full.png")
    dmumu_collapsed_png = joinpath(science_dir, "dmumu_tau_average_full.png")
    return (
        cache_dir = cache_dir,
        science_dir = science_dir,
        cache_h5 = cache_h5,
        combined_h5 = combined_h5,
        delta_png = delta_png,
        dmumu_heatmap_png = dmumu_heatmap_png,
        dmumu_collapsed_png = dmumu_collapsed_png,
    )
end

function estimate_mu_cache_bytes(n_particles::Integer, nsave::Integer, ::Type{T}) where {T}
    return Int128(n_particles) * Int128(nsave) * Int128(sizeof(T))
end

function create_mu_cache_writer(path::AbstractString, n_particles::Integer, nsave::Integer, ::Type{Tout}) where {Tout<:AbstractFloat}
    file = h5open(path, "w")
    mu = create_dataset(file, "mu", Tout, (n_particles, nsave))
    return (file=file, mu=mu, outtype=Tout)
end

function write_mu_cache_step!(writer, save_idx::Integer, mu_step)
    writer.mu[:, save_idx] = writer.outtype.(mu_step)
    return nothing
end

function finalize_mu_cache_writer!(writer, cfg, energy_GeV, t_norm_save, t_s_save, alive_fraction_save, dt, Omega0, B0)
    file = writer.file
    file["t_norm"] = Float64.(t_norm_save)
    file["t_s"] = Float64.(t_s_save)
    file["alive_fraction"] = Float64.(alive_fraction_save)
    file["energy_GeV"] = Float64[energy_GeV]
    file["dt_s"] = Float64[dt]
    file["Omega0"] = Float64[Omega0]
    file["B0_T"] = Float64[B0]
    file["n_particles"] = Int[cfg[:n_particles]]
    file["trajectory_time_stride"] = Int[cfg[:trajectory_time_stride]]
    file["boundary_mode"] = string(cfg[:boundary])
    file["cache_mode"] = "mu"
    file["cache_output_precision"] = string(writer.outtype)
    file["mu_unit"] = "dimensionless"
    close(file)
    return nothing
end

function reconstruct_mu_step_kernel!(mu_step, x1_step, x2_step, x3_step, p1_step, p2_step, p3_step, Bx, By, Bz, xmin, ymin, zmin, xmax, ymax, zmax, dx, dy, dz, nx, ny, nz)
    particle = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nparticles = length(mu_step)
    particle > nparticles && return

    T = eltype(mu_step)
    xx = x1_step[particle]
    yy = x2_step[particle]
    zz = x3_step[particle]
    px = p1_step[particle]
    py = p2_step[particle]
    pz = p3_step[particle]

    value = T(NaN)
    if !(isnan(xx) || isnan(yy) || isnan(zz) || isnan(px) || isnan(py) || isnan(pz))
        if xmin <= xx <= xmax && ymin <= yy <= ymax && zmin <= zz <= zmax
            bx = CombinedFullMod.trilinear_sample(Bx, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            by = CombinedFullMod.trilinear_sample(By, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            bz = CombinedFullMod.trilinear_sample(Bz, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            value = CombinedFullMod.pitch_mu_from_pB(px, py, pz, bx, by, bz)
        end
    end

    mu_step[particle] = value
    return
end

function run_mu_cache_ensemble(cfg, fields, output_h5::AbstractString; energy_GeV)
    T = cfg[:precision]
    Bx, By, Bz, vx, vy, vz = fields
    nx, ny, nz = size(Bx)
    x, y, z = RunnerMod.build_uniform_coords(cfg, nx, ny, nz, T)

    boundary = cfg[:boundary]
    if !(boundary == :open || boundary == :periodic)
        error("Unknown boundary mode: " * string(boundary) * ". Use :open or :periodic.")
    end
    periodic_boundary = boundary == :periodic

    trajectory_stride = Int(cfg[:trajectory_time_stride])
    cache_out_type = cfg[:mu_cache_output_precision]

    B0 = mean(sqrt.(Bx .^ 2 .+ By .^ 2 .+ Bz .^ 2))
    gamma0 = RunnerMod.energy_to_gamma(energy_GeV, T)
    Omega0 = T(RunnerMod.Q_E) * B0 / (gamma0 * T(RunnerMod.M_P))
    v0 = RunnerMod.energy_to_speed(energy_GeV, T)
    dt_gyro = T(cfg[:eta]) / Omega0
    dt = dt_gyro
    if cfg[:use_cfl]
        dt = min(dt, T(cfg[:cfl]) * RunnerMod.estimate_min_dx(x, y, z) / v0)
    end

    t_end = T(cfg[:tOmega0_max]) / Omega0
    nsteps = Int(floor(t_end / dt)) + 1
    t_s = dt .* collect(T, 0:(nsteps - 1))
    t_norm = t_s .* Omega0
    save_indices = RunnerMod.sampled_step_indices(nsteps, trajectory_stride)
    nsave = length(save_indices)

    x1, x2, x3, p1, p2, p3 = RunnerMod.init_particles(cfg, x, y, z, gamma0, v0, T)
    dx1 = CuArray(x1)
    dx2 = CuArray(x2)
    dx3 = CuArray(x3)
    dp1 = CuArray(p1)
    dp2 = CuArray(p2)
    dp3 = CuArray(p3)
    dalive = CUDA.fill(true, cfg[:n_particles])

    dBx = CuArray(Bx)
    dBy = CuArray(By)
    dBz = CuArray(Bz)
    dvx = CuArray(vx)
    dvy = CuArray(vy)
    dvz = CuArray(vz)

    xmin, xmax = x[1], x[end]
    ymin, ymax = y[1], y[end]
    zmin, zmax = z[1], z[end]
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]

    dx1_step = CUDA.fill(T(NaN), cfg[:n_particles])
    dx2_step = CUDA.fill(T(NaN), cfg[:n_particles])
    dx3_step = CUDA.fill(T(NaN), cfg[:n_particles])
    dp1_step = CUDA.fill(T(NaN), cfg[:n_particles])
    dp2_step = CUDA.fill(T(NaN), cfg[:n_particles])
    dp3_step = CUDA.fill(T(NaN), cfg[:n_particles])
    dmu_step = CUDA.fill(T(NaN), cfg[:n_particles])

    alive_fraction = Vector{Float64}(undef, nsteps)
    mu_cache_gib = RunnerMod.bytes_to_gib(estimate_mu_cache_bytes(cfg[:n_particles], nsave, cache_out_type))

    threads = 256
    blocks = cld(cfg[:n_particles], threads)

    println("  cache mode     = mu")
    println("  save stride    = ", trajectory_stride)
    println("  save steps     = ", nsave, "/", nsteps)
    println("  output type    = ", cache_out_type)
    println("  est. size [GiB]= ", mu_cache_gib)

    elapsed = NaN
    writer = nothing
    try
        writer = create_mu_cache_writer(output_h5, cfg[:n_particles], nsave, cache_out_type)

        elapsed = @elapsed begin
            next_save_ptr = 1
            for si in 1:nsteps
                if (si - 1) % cfg[:progress_every] == 0
                    println("Energy ", energy_GeV, " GeV: step ", si, "/", nsteps)
                end

                do_push = si < nsteps
                @cuda threads=threads blocks=blocks RunnerMod.advance_particles_kernel!(
                    dx1_step, dx2_step, dx3_step, dp1_step, dp2_step, dp3_step,
                    dx1, dx2, dx3, dp1, dp2, dp3, dalive,
                    dBx, dBy, dBz, dvx, dvy, dvz,
                    dt, xmin, ymin, zmin, xmax, ymax, zmax, dx, dy, dz, nx, ny, nz, do_push, periodic_boundary
                )

                alive_fraction[si] = Float64(sum(dalive)) / Float64(cfg[:n_particles])

                if next_save_ptr <= nsave && si == save_indices[next_save_ptr]
                    @cuda threads=threads blocks=blocks reconstruct_mu_step_kernel!(
                        dmu_step,
                        dx1_step, dx2_step, dx3_step,
                        dp1_step, dp2_step, dp3_step,
                        dBx, dBy, dBz,
                        xmin, ymin, zmin,
                        xmax, ymax, zmax,
                        dx, dy, dz,
                        nx, ny, nz,
                    )
                    CUDA.synchronize()
                    write_mu_cache_step!(writer, next_save_ptr, Array(dmu_step))
                    next_save_ptr += 1
                end
            end
            CUDA.synchronize()
        end

        finalize_mu_cache_writer!(
            writer,
            cfg,
            energy_GeV,
            t_norm[save_indices],
            t_s[save_indices],
            alive_fraction[save_indices],
            dt,
            Omega0,
            B0,
        )
    catch
        if writer !== nothing
            try
                close(writer.file)
            catch
            end
        end
        rethrow()
    end

    return (
        cache_path = output_h5,
        cache_gib = mu_cache_gib,
        elapsed = elapsed,
        Omega0 = Float64(Omega0),
        dt = Float64(dt),
    )
end

function read_mu_batch(dataset, particle_indices)
    isempty(particle_indices) && error("Cannot read an empty mu batch.")
    if CombinedFullMod.is_contiguous(particle_indices)
        return dataset[first(particle_indices):last(particle_indices), :]
    end

    output = Array{eltype(dataset), 2}(undef, length(particle_indices), size(dataset, 2))
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
        output[output_offset:(output_offset + run_length - 1), :] = dataset[source_first:source_last, :]

        output_offset += run_length
        run_start = run_stop + 1
    end

    return output
end

function append_cache_metadata!(path_h5::AbstractString, cache_mode::Symbol, cache_h5::AbstractString)
    h5open(path_h5, "r+") do file
        file["cache_mode"] = string(cache_mode)
        file["cache_h5"] = string(cache_h5)
        file["cache_dataset"] = cache_mode == :mu ? "mu" : "positions,momenta"
    end
    return nothing
end

function run_combined_from_mu_cache(cfg)
    mkpath(cfg[:output_dir])
    mkpath(dirname(cfg[:output_h5]))
    mkpath(dirname(cfg[:output_delta_png]))
    mkpath(dirname(cfg[:output_heatmap_png]))
    mkpath(dirname(cfg[:output_collapsed_png]))

    mu_edges, mu_centers = CombinedFullMod.build_mu_edges_full(cfg)
    start_mode = CombinedFullMod.dmumu_start_mode(cfg)
    bin_abs = CombinedFullMod.mu_bin_abs(cfg)
    mu_axis_label = CombinedFullMod.mu_bin_axis_label(start_mode, bin_abs)
    heatmap_title = CombinedFullMod.dmumu_plot_title(bin_abs)
    collapsed_title = CombinedFullMod.dmumu_tau_average_title(bin_abs)

    h5open(cfg[:cache_h5], "r") do cache_file
        mu_dataset = cache_file["mu"]
        t_s = Float64.(read(cache_file["t_s"]))
        t_norm = Float64.(read(cache_file["t_norm"]))
        nsteps = size(mu_dataset, 2)
        total_particles = size(mu_dataset, 1)

        particle_indices = CombinedFullMod.build_particle_indices(total_particles, cfg)
        first_particle = first(particle_indices)
        last_particle = last(particle_indices)
        selected_particle_count = length(particle_indices)
        selected_particle_count > 0 || error("No particles selected.")

        lag_steps = CombinedFullMod.build_selected_lag_steps(nsteps, cfg)
        tau_s = [Float64(t_s[lag_step + 1] - t_s[1]) for lag_step in lag_steps]
        tau_norm = [Float64(t_norm[lag_step + 1] - t_norm[1]) for lag_step in lag_steps]
        estimated_pair_visits = CombinedFullMod.guard_exact_pair_cost!(nsteps, selected_particle_count, lag_steps, cfg)

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

        println("Mu cache HDF5: ", cfg[:cache_h5])
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
        println("Mu bin coordinate: ", CombinedFullMod.mu_bin_coordinate_name(start_mode, bin_abs))
        println("Mu bins: ", n_bins, " from ", first(mu_edges), " to ", last(mu_edges))
        println("Particle chunk: ", chunk_size)
        println("Chunks: ", nchunks)
        println("D_mumu accumulation backend: cpu")
        println("Julia threads: ", Threads.nthreads(), " active pool, max thread id ", Threads.maxthreadid())

        for chunk_id in 1:nchunks
            selection_first = (chunk_id - 1) * chunk_size + 1
            selection_last = min(selected_particle_count, selection_first + chunk_size - 1)
            chunk_indices = particle_indices[selection_first:selection_last]
            println("Chunk ", chunk_id, "/", nchunks, ": ", length(chunk_indices), " selected particles, index span ", first(chunk_indices), "-", last(chunk_indices))

            mu_batch = read_mu_batch(mu_dataset, chunk_indices)
            mu_chunk = permutedims(mu_batch, (2, 1))

            CombinedFullMod.process_mu_chunk_dmumu_cpu!(
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

        delta_df = CombinedFullMod.build_output_dataframe(lag_steps, t_s, t_norm, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)

        mean_delta,
        mean_delta2,
        drift_norm,
        dmumu_raw_norm,
        dmumu_centered_norm,
        dmumu_raw_per_s,
        dmumu_centered_per_s = CombinedFullMod.compute_dmumu_arrays_full(dmumu_counts, dmumu_sum_delta, dmumu_sum_delta2, tau_s, tau_norm, cfg)

        collapsed_raw_norm, collapsed_raw_norm_count_weighted = CombinedFullMod.average_over_tau_full(dmumu_raw_norm, dmumu_counts, cfg)
        collapsed_centered_norm, collapsed_centered_norm_count_weighted = CombinedFullMod.average_over_tau_full(dmumu_centered_norm, dmumu_counts, cfg)

        save_cfg = merge_cfg(cfg, Dict{Symbol, Any}(:trajectory_h5 => cfg[:cache_h5]))
        CombinedFullMod.save_combined_full_h5(
            cfg[:output_h5],
            save_cfg,
            :mu_cache,
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
        append_cache_metadata!(cfg[:output_h5], :mu, cfg[:cache_h5])
        println("Saved combined HDF5 to ", cfg[:output_h5])

        CombinedFullMod.plot_delta_mu2(delta_df, cfg[:output_delta_png]; use_usetex=cfg[:use_usetex])
        println("Saved delta_mu2 plot to ", cfg[:output_delta_png])

        CombinedFullMod.plot_dmumu_heatmap_full(
            cfg[:output_heatmap_png],
            mu_edges,
            tau_norm,
            dmumu_centered_norm;
            use_usetex=cfg[:use_usetex],
            mu_axis_label=mu_axis_label,
            title_label=heatmap_title,
        )
        println("Saved D_mumu heatmap to ", cfg[:output_heatmap_png])

        CombinedFullMod.plot_collapsed_dmumu_full(
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

    return nothing
end

function run_energy_pipeline!(cfg, fields, base_runner_cfg, energy_GeV)
    paths = build_energy_paths(cfg, energy_GeV)
    mkpath(paths.cache_dir)
    mkpath(paths.science_dir)
    requested_analysis_cfg = merge_cfg(CombinedFullMod.COMBINED_FULL_CFG, cfg[:combined_overrides])

    println()
    println("=== Energy ", energy_GeV, " GeV ===")
    if !cfg[:compute_dmumu] && cfg[:skip_completed_outputs] && isfile(paths.cache_h5)
        verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
        println("Cache already verified; D_mumu disabled for energy ", energy_GeV, " GeV")
        return summary_row(
            energy_GeV,
            "skipped_existing_cache",
            false,
            paths.cache_h5,
            "",
            "verified existing cache; D_mumu disabled",
        )
    end

    if cfg[:skip_completed_outputs] && compact_outputs_complete(paths, requested_analysis_cfg)
        deleted = false
        if isfile(paths.cache_h5)
            verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
            deleted = delete_cache_if_requested(paths.cache_h5, cfg)
        end
        println("Combined outputs already verified; skipping energy ", energy_GeV, " GeV")
        return summary_row(
            energy_GeV,
            "skipped_existing_outputs",
            deleted,
            paths.cache_h5,
            paths.combined_h5,
            "verified existing combined outputs",
        )
    end

    if cfg[:reuse_existing_cache] && isfile(paths.cache_h5)
        println("Reusing existing cache file ", paths.cache_h5)
        verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
    else
        if cfg[:cache_mode] == :phase_space
            runner_cfg = merge_cfg(base_runner_cfg, Dict{Symbol, Any}(
                :output_dir => paths.cache_dir,
                :energies => [energy_GeV],
            ))
            runner_result = RunnerMod.run_gpu_ensemble(runner_cfg, fields; energy_GeV=energy_GeV)
            verify_phase_space_cache_h5(runner_result.phase_space_path)
        elseif cfg[:cache_mode] == :mu
            run_mu_cache_ensemble(base_runner_cfg, fields, paths.cache_h5; energy_GeV=energy_GeV)
            verify_mu_cache_h5(paths.cache_h5)
        else
            error("Unknown cache mode: " * string(cfg[:cache_mode]))
        end
    end

    if !cfg[:compute_dmumu]
        verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
        println("D_mumu disabled; keeping verified cache file ", paths.cache_h5)
        return summary_row(
            energy_GeV,
            "cache_only",
            false,
            paths.cache_h5,
            "",
            "verified cache; D_mumu disabled",
        )
    end

    analysis_cfg = merge_cfg(
        requested_analysis_cfg,
        Dict{Symbol, Any}(
            :trajectory_h5 => paths.cache_h5,
            :cache_h5 => paths.cache_h5,
            :cache_mode => cfg[:cache_mode],
            :turbulence_h5 => cfg[:turbulence_h5],
            :output_dir => paths.science_dir,
            :output_h5 => paths.combined_h5,
            :output_delta_png => paths.delta_png,
            :output_heatmap_png => paths.dmumu_heatmap_png,
            :output_collapsed_png => paths.dmumu_collapsed_png,
        ),
    )

    if cfg[:cache_mode] == :phase_space
        CombinedFullMod.run_combined_full(analysis_cfg)
        append_cache_metadata!(paths.combined_h5, :phase_space, paths.cache_h5)
    else
        run_combined_from_mu_cache(analysis_cfg)
    end

    verify_combined_outputs(paths.combined_h5, paths.delta_png, paths.dmumu_heatmap_png, paths.dmumu_collapsed_png)
    deleted = delete_cache_if_requested(paths.cache_h5, cfg)
    return summary_row(
        energy_GeV,
        "ok",
        deleted,
        paths.cache_h5,
        paths.combined_h5,
        "verified combined outputs",
    )
end

function run_campaign(cfg)
    mkpath(cfg[:campaign_root])
    base_runner_cfg = merge_cfg(
        RunnerMod.CFG,
        merge_cfg(
            cfg[:trajectory_overrides],
            Dict{Symbol, Any}(
                :file => cfg[:turbulence_h5],
                :mu_cache_output_precision => cfg[:mu_cache_output_precision],
            ),
        ),
    )

    println("Campaign ", cfg[:campaign_tag])
    println("  campaign root            = ", cfg[:campaign_root])
    println("  turbulence H5            = ", cfg[:turbulence_h5])
    println("  cache mode               = ", cfg[:cache_mode])
    println("Loading turbulence fields once for trajectory generation")
    fields = RunnerMod.load_static_fields(base_runner_cfg, base_runner_cfg[:precision])

    summary_rows = Any[]
    summary_path = joinpath(cfg[:campaign_root], "campaign_summary.tsv")
    for energy_GeV in cfg[:energies]
        try
            push!(summary_rows, run_energy_pipeline!(cfg, fields, base_runner_cfg, energy_GeV))
            write_summary(summary_path, summary_rows)
        catch error_instance
            note = sprint(showerror, error_instance, catch_backtrace())
            paths = build_energy_paths(cfg, energy_GeV)
            push!(summary_rows, summary_row(
                energy_GeV,
                "error",
                false,
                paths.cache_h5,
                paths.combined_h5,
                note,
            ))
            write_summary(summary_path, summary_rows)
            if cfg[:stop_on_error]
                rethrow()
            end
        end
    end

    write_summary(summary_path, summary_rows)
    println()
    println("Saved campaign summary to ", summary_path)
    return summary_rows
end

function materialize_campaign_cfg(cfg, campaign_spec)
    campaign_cfg = Dict{Symbol, Any}(
        :campaign_tag => campaign_spec[:campaign_tag],
        :campaign_root => campaign_spec[:campaign_root],
        :turbulence_h5 => campaign_spec[:turbulence_h5],
        :cache_mode => cfg[:cache_mode],
        :compute_dmumu => cfg[:compute_dmumu],
        :energies => get(campaign_spec, :energies, cfg[:energies]),
        :delete_cache_on_success => cfg[:delete_cache_on_success],
        :reuse_existing_cache => cfg[:reuse_existing_cache],
        :skip_completed_outputs => cfg[:skip_completed_outputs],
        :stop_on_error => cfg[:stop_on_error],
        :run_all_particles_dmumu => cfg[:run_all_particles_dmumu],
        :mu_cache_output_precision => cfg[:mu_cache_output_precision],
        :trajectory_overrides => shallow_copy_dict(cfg[:trajectory_overrides]),
        :combined_overrides => shallow_copy_dict(cfg[:combined_overrides]),
    )
    return campaign_cfg
end

function resolve_repo_path(value)
    path = expanduser(String(value))
    return isabspath(path) ? normpath(path) : normpath(joinpath(PIPELINE_ROOT, path))
end

function resolve_cli_path(value)
    path = expanduser(String(value))
    return isabspath(path) ? normpath(path) : abspath(path)
end

function config_path_from_args(args)
    for argument in args
        startswith(argument, "--config=") && return split(argument, "=", limit=2)[2]
    end
    return nothing
end

function toml_section(config, name::AbstractString)
    section = get(config, name, nothing)
    section === nothing && return Dict{String, Any}()
    section isa AbstractDict || error("[" * name * "] must be a TOML table.")
    return section
end

function validate_toml_keys(section, allowed_keys, section_name::AbstractString)
    unknown = sort([String(key) for key in keys(section) if !(String(key) in allowed_keys)])
    isempty(unknown) && return nothing
    error("Unknown key(s) in " * section_name * ": " * join(unknown, ", "))
end

function config_bool(value, key_name::AbstractString)
    value isa Bool && return value
    if value isa AbstractString
        normalized = lowercase(strip(value))
        normalized in ("true", "yes", "1") && return true
        normalized in ("false", "no", "0") && return false
    end
    error(key_name * " must be true or false.")
end

function config_int(value, key_name::AbstractString)
    value isa Integer && return Int(value)
    if value isa AbstractFloat && isinteger(value)
        return Int(value)
    elseif value isa AbstractString
        return parse(Int, strip(value))
    end
    error(key_name * " must be an integer.")
end

function config_float(value, key_name::AbstractString)
    value isa Real && return Float64(value)
    value isa AbstractString && return parse(Float64, strip(value))
    error(key_name * " must be numeric.")
end

function config_precision(value, key_name::AbstractString)
    normalized = lowercase(strip(String(value)))
    normalized == "float32" && return Float32
    normalized == "float64" && return Float64
    error(key_name * " must be Float32 or Float64.")
end

function config_symbol(value, key_name::AbstractString)
    value isa AbstractString || error(key_name * " must be a string.")
    return Symbol(replace(lowercase(strip(value)), "-" => "_"))
end

function config_string_selector(value, key_name::AbstractString)
    if value isa AbstractString
        return split_csv_selector(value; flag_name=key_name)
    elseif value isa AbstractVector
        isempty(value) && error(key_name * " cannot be an empty array. Use \"all\" or omit the key.")
        values = String[strip(String(item)) for item in value]
        any(isempty, values) && error(key_name * " cannot contain empty values.")
        return values
    end
    error(key_name * " must be a string or array of strings.")
end

function config_campaign_selector(value, key_name::AbstractString)
    selected = config_string_selector(value, key_name)
    selected === nothing && return nothing
    return normalize_campaign_selector.(selected)
end

function config_energy_values(value, key_name::AbstractString)
    if value isa AbstractString
        selected = split_csv_selector(value; flag_name=key_name)
        selected === nothing && error(key_name * " must list one or more energies.")
        return [parse(Float64, energy) for energy in selected]
    elseif value isa AbstractVector
        isempty(value) && error(key_name * " cannot be empty.")
        return [config_float(energy, key_name) for energy in value]
    end
    error(key_name * " must be a string or array of numbers.")
end

function config_tuple3_strings(value, key_name::AbstractString)
    values = if value isa AbstractString
        split_csv_selector(value; flag_name=key_name)
    elseif value isa AbstractVector
        String[strip(String(item)) for item in value]
    else
        error(key_name * " must be a string or array of three strings.")
    end
    values === nothing && error(key_name * " must contain three dataset paths.")
    length(values) == 3 || error(key_name * " must contain exactly three dataset paths.")
    any(isempty, values) && error(key_name * " cannot contain empty dataset paths.")
    return Tuple(values)
end

function config_field_subset(value, key_name::AbstractString)
    if value isa AbstractString
        normalized = lowercase(strip(value))
        normalized in ("", "none", "nothing") && return nothing
        parts = split(value, ",")
        length(parts) == 3 || error(key_name * " must be none or three integers.")
        return Tuple([parse(Int, strip(part)) for part in parts])
    elseif value isa AbstractVector
        isempty(value) && return nothing
        length(value) == 3 || error(key_name * " must be an empty array or three integers.")
        return Tuple([config_int(item, key_name) for item in value])
    end
    error(key_name * " must be none, an empty array, or three integers.")
end

function config_maybe_int(value, key_name::AbstractString)
    if value isa AbstractString && (lowercase(strip(value)) in ("none", "nothing"))
        return nothing
    end
    return config_int(value, key_name)
end

function config_maybe_particle_count(value, key_name::AbstractString)
    if value isa AbstractString && (lowercase(strip(value)) in ("all", "none", "nothing"))
        return nothing
    end
    return config_int(value, key_name)
end

function convert_override_value(key::Symbol, value, key_name::AbstractString)
    key in (:B_paths, :v_paths) && return config_tuple3_strings(value, key_name)
    key == :field_subset && return config_field_subset(value, key_name)
    key in (:precision, :trajectory_output_precision, :compute_precision) && return config_precision(value, key_name)
    key in (:boundary, :compute_backend, :particle_selection) && return config_symbol(value, key_name)
    key == :lag_mode && return CombinedFullMod.parse_lag_mode(String(value))
    key == :dmumu_start_mode && return CombinedFullMod.parse_dmumu_start_mode(value)
    key == :mu_bin_abs && return config_bool(value, key_name)
    key == :n_particles_to_use && return config_maybe_particle_count(value, key_name)
    key == :max_lag_steps && return config_maybe_int(value, key_name)
    return value
end

function apply_override_section!(target, section, section_name::AbstractString)
    allowed_keys = Set(string(key) for key in keys(target))
    validate_toml_keys(section, allowed_keys, section_name)
    for (key, value) in section
        target_key = Symbol(key)
        target[target_key] = convert_override_value(target_key, value, section_name * "." * String(key))
    end
    return nothing
end

function config_available_modes(value, key_name::AbstractString)
    selected = config_string_selector(value, key_name)
    selected === nothing && return String.(MODE_NAMES)
    return selected
end

function config_dmumu_particles(value, key_name::AbstractString)
    normalized = replace(lowercase(strip(String(value))), "-" => "_")
    normalized in ("sample", "sampled", "reduced") && return :sample
    normalized in ("all", "all_particles") && return :all
    error(key_name * " must be sample or all.")
end

function config_campaign_requests(value, key_name::AbstractString)
    value isa AbstractVector || error(key_name * " must be an array of tables.")
    isempty(value) && error(key_name * " cannot be empty.")

    requests = Dict{Symbol, Any}[]
    for (index, item) in enumerate(value)
        item isa AbstractDict || error(key_name * " entries must be TOML tables.")
        entry_name = key_name * "[" * string(index) * "]"
        validate_toml_keys(item, Set(["mode", "campaign", "energies_gev", "energy_gev"]), entry_name)

        request = Dict{Symbol, Any}()
        haskey(item, "mode") && (request[:mode_name] = strip(String(item["mode"])))
        haskey(item, "campaign") && (request[:campaign_tag] = normalize_campaign_selector(String(item["campaign"])))
        isempty(request) && error(entry_name * " must define mode or campaign.")

        if haskey(item, "energies_gev")
            request[:energies] = config_energy_values(item["energies_gev"], entry_name * ".energies_gev")
        elseif haskey(item, "energy_gev")
            request[:energies] = config_energy_values([item["energy_gev"]], entry_name * ".energy_gev")
        end

        push!(requests, request)
    end
    return requests
end

function apply_input_section!(cfg, input)
    legacy_keys = Set(["layout", "mode_decomposition_root", "mhd512_total_h5", "mhd512_mode_dir"])
    generic_keys = Set(["h5_dir", "label", "medium", "file_stem", "total_file", "mode_file_pattern"])
    allowed_keys = union(legacy_keys, generic_keys)
    validate_toml_keys(input, allowed_keys, "[input]")

    has_legacy = any(key -> haskey(input, key), legacy_keys)
    has_generic = any(key -> haskey(input, key), generic_keys)
    has_legacy && has_generic && error("[input] cannot mix legacy layout keys with generic h5_dir/label keys.")

    if has_generic
        cfg[:input_kind] = :generic
        haskey(input, "h5_dir") && (cfg[:input_spec][:h5_dir] = resolve_repo_path(input["h5_dir"]))
        haskey(input, "label") && (cfg[:input_spec][:label] = strip(String(input["label"])))
        haskey(input, "medium") && (cfg[:input_spec][:medium] = normalize_medium(input["medium"]))
        haskey(input, "file_stem") && (cfg[:input_spec][:file_stem] = strip(String(input["file_stem"])))
        haskey(input, "total_file") && (cfg[:input_spec][:total_file] = strip(String(input["total_file"])))
        haskey(input, "mode_file_pattern") && (cfg[:input_spec][:mode_file_pattern] = strip(String(input["mode_file_pattern"])))
        return nothing
    end

    haskey(input, "layout") && (cfg[:input_layout] = parse_input_layout(input["layout"]))
    haskey(input, "mode_decomposition_root") && (cfg[:input_paths][:mode_decomposition_root] = resolve_repo_path(input["mode_decomposition_root"]))
    haskey(input, "mhd512_total_h5") && (cfg[:input_paths][:mhd512_total_h5] = resolve_repo_path(input["mhd512_total_h5"]))
    haskey(input, "mhd512_mode_dir") && (cfg[:input_paths][:mhd512_mode_dir] = resolve_repo_path(input["mhd512_mode_dir"]))
    return nothing
end

function apply_dmumu_section!(cfg, section)
    allowed_keys = union(Set(string(key) for key in keys(cfg[:combined_overrides])), Set(["particles"]))
    validate_toml_keys(section, allowed_keys, "[dmumu]")

    override_section = Dict{String, Any}(String(key) => value for (key, value) in section if String(key) != "particles")
    apply_override_section!(cfg[:combined_overrides], override_section, "[dmumu]")

    haskey(section, "particles") && (cfg[:dmumu_particles] = config_dmumu_particles(section["particles"], "[dmumu].particles"))
    return nothing
end

function apply_toml_config!(cfg, config_path::AbstractString)
    absolute_config_path = resolve_cli_path(config_path)
    isfile(absolute_config_path) || error("Config file not found: " * absolute_config_path)
    config = TOML.parsefile(absolute_config_path)

    validate_toml_keys(config, Set(["input", "output", "run", "particles", "dmumu", "trajectory", "combined"]), "config")

    input = toml_section(config, "input")
    apply_input_section!(cfg, input)

    output = toml_section(config, "output")
    validate_toml_keys(output, Set(["root"]), "[output]")
    haskey(output, "root") && (cfg[:output_root] = resolve_repo_path(output["root"]))

    run = toml_section(config, "run")
    validate_toml_keys(
        run,
        Set([
            "cache_mode",
            "compute_dmumu",
            "mode_decomposition_available",
            "available_modes",
            "energies_gev",
            "turbulence",
            "modes",
            "mode",
            "campaign",
            "campaigns",
            "delete_cache_on_success",
            "reuse_existing_cache",
            "skip_completed_outputs",
            "stop_on_error",
            "all_particles_dmumu",
            "mu_cache_output_precision",
            "smoke",
        ]),
        "[run]",
    )
    haskey(run, "cache_mode") && (cfg[:cache_mode] = parse_cache_mode(run["cache_mode"]))
    haskey(run, "compute_dmumu") && (cfg[:compute_dmumu] = config_bool(run["compute_dmumu"], "[run].compute_dmumu"))
    haskey(run, "mode_decomposition_available") && (cfg[:mode_decomposition_available] = config_bool(run["mode_decomposition_available"], "[run].mode_decomposition_available"))
    haskey(run, "available_modes") && (cfg[:available_modes] = config_available_modes(run["available_modes"], "[run].available_modes"))
    haskey(run, "energies_gev") && (cfg[:energies] = config_energy_values(run["energies_gev"], "[run].energies_gev"))
    haskey(run, "turbulence") && (cfg[:selected_turbulences] = config_string_selector(run["turbulence"], "[run].turbulence"))
    haskey(run, "modes") && (cfg[:selected_modes] = config_string_selector(run["modes"], "[run].modes"))
    haskey(run, "mode") && (cfg[:selected_modes] = config_string_selector(run["mode"], "[run].mode"))
    haskey(run, "campaign") && (cfg[:selected_campaigns] = config_campaign_selector(run["campaign"], "[run].campaign"))
    haskey(run, "campaigns") && (cfg[:campaign_requests] = config_campaign_requests(run["campaigns"], "[run].campaigns"))
    haskey(run, "delete_cache_on_success") && (cfg[:delete_cache_on_success] = config_bool(run["delete_cache_on_success"], "[run].delete_cache_on_success"))
    haskey(run, "reuse_existing_cache") && (cfg[:reuse_existing_cache] = config_bool(run["reuse_existing_cache"], "[run].reuse_existing_cache"))
    haskey(run, "skip_completed_outputs") && (cfg[:skip_completed_outputs] = config_bool(run["skip_completed_outputs"], "[run].skip_completed_outputs"))
    haskey(run, "stop_on_error") && (cfg[:stop_on_error] = config_bool(run["stop_on_error"], "[run].stop_on_error"))
    haskey(run, "all_particles_dmumu") && config_bool(run["all_particles_dmumu"], "[run].all_particles_dmumu") && (cfg[:dmumu_particles] = :all)
    haskey(run, "mu_cache_output_precision") && (cfg[:mu_cache_output_precision] = config_precision(run["mu_cache_output_precision"], "[run].mu_cache_output_precision"))
    haskey(run, "smoke") && (cfg[:smoke] = config_bool(run["smoke"], "[run].smoke"))

    apply_override_section!(cfg[:trajectory_overrides], toml_section(config, "trajectory"), "[trajectory]")
    apply_override_section!(cfg[:trajectory_overrides], toml_section(config, "particles"), "[particles]")
    apply_override_section!(cfg[:combined_overrides], toml_section(config, "combined"), "[combined]")
    apply_dmumu_section!(cfg, toml_section(config, "dmumu"))
    return absolute_config_path
end

function apply_all_particles_dmumu!(cfg)
    cfg[:run_all_particles_dmumu] = true
    merge!(cfg[:combined_overrides], Dict{Symbol, Any}(
        :first_particle => 1,
        :n_particles_to_use => nothing,
        :particle_selection => :range,
        :allow_huge => true,
    ))
    return nothing
end

function runtime_config()
    cfg = shallow_copy_dict(CACHE_PIPELINE_CFG)
    cfg[:trajectory_overrides] = shallow_copy_dict(CACHE_PIPELINE_CFG[:trajectory_overrides])
    cfg[:combined_overrides] = shallow_copy_dict(CACHE_PIPELINE_CFG[:combined_overrides])
    cfg[:input_kind] = :legacy
    cfg[:input_spec] = default_input_spec()
    cfg[:input_paths] = default_input_paths()
    cfg[:output_root] = joinpath(PIPELINE_ROOT, "outputs", "campaigns_cache")
    cfg[:input_layout] = :mp_weakb
    cfg[:mode_decomposition_available] = false
    cfg[:available_modes] = String.(MODE_NAMES)
    cfg[:dmumu_particles] = :sample
    cfg[:smoke] = false

    config_path = config_path_from_args(ARGS)
    cfg[:config_path] = config_path === nothing ? nothing : apply_toml_config!(cfg, config_path)

    cache_mode = cfg[:cache_mode]
    input_layout = cfg[:input_layout]
    for argument in ARGS
        if startswith(argument, "--cache-mode=")
            cache_mode = parse_cache_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--cache=")
            cache_mode = parse_cache_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--layout=")
            input_layout = parse_input_layout(split(argument, "=", limit=2)[2])
            cfg[:input_kind] = :legacy
        elseif startswith(argument, "--input-layout=")
            input_layout = parse_input_layout(split(argument, "=", limit=2)[2])
            cfg[:input_kind] = :legacy
        elseif argument == "--mhd512"
            input_layout = :mhd512
            cfg[:input_kind] = :legacy
        elseif argument == "--mp-weakb"
            input_layout = :mp_weakb
            cfg[:input_kind] = :legacy
        elseif startswith(argument, "--output-root=")
            cfg[:output_root] = resolve_cli_path(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--input-h5-dir=")
            cfg[:input_kind] = :generic
            cfg[:input_spec][:h5_dir] = resolve_cli_path(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--input-label=")
            cfg[:input_kind] = :generic
            cfg[:input_spec][:label] = strip(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--input-medium=")
            cfg[:input_kind] = :generic
            cfg[:input_spec][:medium] = normalize_medium(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--file-stem=")
            cfg[:input_kind] = :generic
            cfg[:input_spec][:file_stem] = strip(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--total-file=")
            cfg[:input_kind] = :generic
            cfg[:input_spec][:total_file] = strip(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--mode-file-pattern=")
            cfg[:input_kind] = :generic
            cfg[:input_spec][:mode_file_pattern] = strip(split(argument, "=", limit=2)[2])
        elseif argument == "--mode-decomposition-available"
            cfg[:mode_decomposition_available] = true
        elseif argument == "--no-mode-decomposition"
            cfg[:mode_decomposition_available] = false
        elseif argument == "--compute-dmumu"
            cfg[:compute_dmumu] = true
        elseif argument == "--no-compute-dmumu"
            cfg[:compute_dmumu] = false
        elseif startswith(argument, "--mode-decomposition-root=")
            cfg[:input_kind] = :legacy
            cfg[:input_paths][:mode_decomposition_root] = resolve_cli_path(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--mhd512-total-h5=")
            cfg[:input_kind] = :legacy
            cfg[:input_paths][:mhd512_total_h5] = resolve_cli_path(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--mhd512-mode-dir=")
            cfg[:input_kind] = :legacy
            cfg[:input_paths][:mhd512_mode_dir] = resolve_cli_path(split(argument, "=", limit=2)[2])
        end
    end
    cfg[:cache_mode] = cache_mode
    cfg[:input_layout] = input_layout

    smoke_run = cfg[:smoke] || any(argument -> argument == "--smoke", ARGS)
    cfg[:smoke] = smoke_run

    root_prefix = joinpath(cfg[:output_root], cache_mode_label(cache_mode))
    if smoke_run
        root_prefix = joinpath(cfg[:output_root], "smoke", cache_mode_label(cache_mode))
    end
    all_campaigns = build_campaigns(cfg, root_prefix)

    selected_turbulences = get(cfg, :selected_turbulences, nothing)
    selected_modes = get(cfg, :selected_modes, nothing)
    selected_campaigns = get(cfg, :selected_campaigns, nothing)
    campaign_requests = get(cfg, :campaign_requests, nothing)

    if smoke_run
        cfg[:energies] = [1e5, 3e5]
        selected_turbulences = nothing
        selected_modes = nothing
        selected_campaigns = [first(all_campaigns)[:campaign_tag]]
        campaign_requests = nothing
        merge!(cfg[:trajectory_overrides], Dict{Symbol, Any}(
            :n_particles => 64,
            :precision => Float32,
            :field_subset => (16, 16, 16),
            :tOmega0_max => 20.0,
            :trajectory_time_stride => 1,
            :progress_every => 20,
        ))
        merge!(cfg[:combined_overrides], Dict{Symbol, Any}(
            :field_subset => (16, 16, 16),
            :particle_chunk_size => 32,
            :first_particle => 1,
            :n_particles_to_use => 64,
            :particle_selection => :range,
            :dmumu_start_mode => :sliding,
            :min_lag_steps => 1,
            :n_lag_samples => 8,
            :max_lag_steps => 20,
            :n_mu_bins => 8,
            :min_count_per_cell => 3,
        ))
    end

    cli_requested_campaign_selection = false
    for argument in ARGS
        if startswith(argument, "--turbulence=")
            selected_turbulences = split_csv_selector(split(argument, "=", limit=2)[2]; flag_name="--turbulence")
            cli_requested_campaign_selection = true
        elseif startswith(argument, "--mode=")
            selected_modes = split_csv_selector(split(argument, "=", limit=2)[2]; flag_name="--mode")
            cli_requested_campaign_selection = true
        elseif startswith(argument, "--modes=")
            selected_modes = split_csv_selector(split(argument, "=", limit=2)[2]; flag_name="--modes")
            cli_requested_campaign_selection = true
        elseif startswith(argument, "--campaign=")
            selected_campaigns = parse_campaign_selector(split(argument, "=", limit=2)[2])
            cli_requested_campaign_selection = true
        elseif startswith(argument, "--energy=")
            cfg[:energies] = config_energy_values(split(argument, "=", limit=2)[2], "--energy")
        elseif startswith(argument, "--energies=")
            cfg[:energies] = config_energy_values(split(argument, "=", limit=2)[2], "--energies")
        elseif startswith(argument, "--dmumu-start-mode=")
            cfg[:combined_overrides][:dmumu_start_mode] = CombinedFullMod.parse_dmumu_start_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--min-lag-steps=")
            cfg[:combined_overrides][:min_lag_steps] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-mode=")
            cfg[:combined_overrides][:lag_mode] = CombinedFullMod.parse_lag_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--n-lag-samples=")
            cfg[:combined_overrides][:n_lag_samples] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--lag-step-stride=")
            cfg[:combined_overrides][:lag_step_stride] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--max-lag-steps=")
            cfg[:combined_overrides][:max_lag_steps] = config_maybe_int(split(argument, "=", limit=2)[2], "--max-lag-steps")
        elseif startswith(argument, "--n-mu-bins=")
            cfg[:combined_overrides][:n_mu_bins] = parse(Int, split(argument, "=", limit=2)[2])
        elseif argument == "--mu-bin-abs"
            cfg[:combined_overrides][:mu_bin_abs] = true
        elseif argument == "--no-mu-bin-abs" || argument == "--signed-mu-bins"
            cfg[:combined_overrides][:mu_bin_abs] = false
        elseif startswith(argument, "--mu-bin-abs=")
            cfg[:combined_overrides][:mu_bin_abs] = config_bool(split(argument, "=", limit=2)[2], "--mu-bin-abs")
        elseif startswith(argument, "--mu-min=")
            cfg[:combined_overrides][:mu_min] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--mu-max=")
            cfg[:combined_overrides][:mu_max] = parse(Float64, split(argument, "=", limit=2)[2])
        end
    end
    cli_requested_campaign_selection && (campaign_requests = nothing)

    if any(argument -> argument == "--keep-trajectories", ARGS) || any(argument -> argument == "--keep-caches", ARGS)
        cfg[:delete_cache_on_success] = false
    end
    if any(argument -> argument == "--regenerate-trajectories", ARGS) || any(argument -> argument == "--regenerate-caches", ARGS)
        cfg[:reuse_existing_cache] = false
    end
    if any(argument -> argument == "--force-recompute", ARGS)
        cfg[:skip_completed_outputs] = false
    end
    if any(argument -> argument == "--all-particles-dmumu", ARGS)
        cfg[:dmumu_particles] = :all
    end
    if cfg[:dmumu_particles] == :all
        apply_all_particles_dmumu!(cfg)
    end

    validate_selected_values(all_campaigns, selected_turbulences, :turbulence_tag, "--turbulence")
    validate_selected_values(all_campaigns, selected_modes, :mode_name, "--mode")
    validate_selected_values(all_campaigns, selected_campaigns, :campaign_tag, "--campaign")

    requested_campaigns = filter_campaigns_by_requests(all_campaigns, campaign_requests, cfg[:energies])
    cfg[:mode_campaigns] = requested_campaigns === nothing ?
        filter_campaigns(all_campaigns, selected_turbulences, selected_modes, selected_campaigns) :
        requested_campaigns
    validate_mode_files(cfg[:mode_campaigns])
    return cfg
end

function run_mode_campaigns(cfg)
    results = Dict{String, Any}()
    for campaign_spec in cfg[:mode_campaigns]
        println()
        println("=== Campaign ", campaign_spec[:campaign_tag], " ===")
        println("  input H5                 = ", campaign_spec[:turbulence_h5])
        campaign_cfg = materialize_campaign_cfg(cfg, campaign_spec)
        results[string(campaign_spec[:campaign_tag])] = run_campaign(campaign_cfg)
    end
    return results
end

function campaign_tags(cfg)
    return [campaign[:campaign_tag] for campaign in cfg[:mode_campaigns]]
end

function main()
    cfg = runtime_config()
    println("Multimode multi-energy cache pipeline")
    cfg[:config_path] !== nothing && println("  config                   = ", cfg[:config_path])
    println("  input kind               = ", cfg[:input_kind])
    if cfg[:input_kind] == :generic
        println("  input medium             = ", cfg[:input_spec][:medium])
        println("  input label              = ", cfg[:input_spec][:label])
        println("  mode decomposition       = ", cfg[:mode_decomposition_available])
    else
        println("  input layout             = ", cfg[:input_layout])
    end
    println("  campaigns                = ", campaign_tags(cfg))
    println("  cache mode               = ", cfg[:cache_mode])
    println("  compute D_mumu           = ", cfg[:compute_dmumu])
    println("  energies [GeV]           = ", cfg[:energies])
    println("  output root              = ", cfg[:output_root])
    println("  delete cache             = ", cfg[:delete_cache_on_success])
    println("  reuse existing cache     = ", cfg[:reuse_existing_cache])
    println("  skip completed outputs   = ", cfg[:skip_completed_outputs])
    println("  stop on error            = ", cfg[:stop_on_error])
    println("  D_mumu particle mode     = ", cfg[:run_all_particles_dmumu] ? "all particles" : "reduced selection")
    run_mode_campaigns(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
