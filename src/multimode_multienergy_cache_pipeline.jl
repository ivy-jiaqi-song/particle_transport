haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

const PIPELINE_ROOT = dirname(@__DIR__)

module RunnerMod
include(joinpath(@__DIR__, "phase_space_gpu_runner.jl"))
end

module CombinedFullMod
include(joinpath(@__DIR__, "compute_delta_mu2_dmumu_full.jl"))
end

module DppFullMod
include(joinpath(@__DIR__, "compute_dpp_full.jl"))
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
    :compute_dpp => false,
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
        :integration_steps_per_gyroperiod => nothing,
        :tOmega0_max => 5000.0,
        :trajectory_duration_gyroperiods => nothing,
        :use_cfl => false,
        :cfl => 0.2,
        :seed => 42,
        :n_particles => 100000,
        :precision => Float64,
        :field_subset => nothing,
        :boundary => :periodic,
        :injection_mode => :isotropic,
        :injection_mu0 => 0.0,
        :injection_position_mode => :random,
        :injection_position => (0.5, 0.5, 0.5),
        :injection_position_unit => :box_fraction,
        :trajectory_time_stride => 1,
        :trajectory_save_interval_gyroperiods => nothing,
        :trajectory_output_precision => Float32,
        :progress_every => 5000,
    ),
    :dmumu_overrides => Dict{Symbol, Any}(
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
        :use_usetex => false,
    ),
    :dpp_overrides => Dict{Symbol, Any}(
        :particle_chunk_size => 128,
        :first_particle => 1,
        :n_particles_to_use => 10000,
        :particle_selection => :block_random,
        :particle_seed => 20260423,
        :particle_block_size => 128,
        :lag_mode => :uniform_samples,
        :lag_min_gyroperiods => nothing,
        :lag_max_gyroperiods => nothing,
        :lag_stride_gyroperiods => nothing,
        :min_lag_steps => 1,
        :n_lag_samples => 40,
        :max_lag_steps => nothing,
        :lag_step_stride => 1,
        :n_energy_snapshots => 5,
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
    input_spec[:total_file] === nothing && error("[input].total_file is required and must point to the total-field HDF5 used for the shared time reference.")
    return input_file_path(input_spec, input_spec[:total_file])
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
    total_h5 = generic_total_h5_path(input_spec)

    campaigns = Dict{Symbol, Any}[]
    push!(
        campaigns,
        Dict{Symbol, Any}(
            :campaign_tag => medium * "/" * label * "/total",
            :campaign_aliases => [label * "/total", label],
            :medium => medium,
            :turbulence_tag => label,
            :mode_name => "total",
            :campaign_root => joinpath(base_root, "total"),
            :turbulence_h5 => total_h5,
            :reference_h5 => total_h5,
        ),
    )
    if mode_decomposition_available
        for mode_name in available_modes
            mode_name = strip(String(mode_name))
            isempty(mode_name) && error("[run].available_modes cannot contain empty mode names.")
            lowercase(mode_name) == "total" && continue
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
                    :reference_h5 => total_h5,
                ),
            )
        end
    end
    return campaigns
end

function build_mode_campaigns(root_prefix::AbstractString, input_paths)
    error("Legacy mp-weakb layout no longer has enough information to resolve the required total-field reference. Use [input].h5_dir and [input].total_file.")
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
            :reference_h5 => input_paths[:mhd512_total_h5],
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
                :reference_h5 => input_paths[:mhd512_total_h5],
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
    append!(missing, [campaign[:reference_h5] for campaign in campaigns if !isfile(campaign[:reference_h5])])
    missing = unique(missing)
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

function verify_file_nonempty(path)
    error("Missing file path for verification: " * string(path))
end

function verify_phase_space_cache_h5(path_h5::AbstractString)
    verify_file_nonempty(path_h5)
    h5open(path_h5, "r") do file
        positions = file["positions"]
        momenta = file["momenta"]
        t_s = read(file["t_s"])
        t_norm = read(file["t_norm"])
        t_gyroperiods = haskey(file, "t_gyroperiods") ? read(file["t_gyroperiods"]) : RunnerMod.t_gyroperiods_from_axes(t_s, t_norm)
        alive_fraction = read(file["alive_fraction"])
        size(positions) == size(momenta) || error("Phase-space verification failed: positions/momenta shapes differ for " * path_h5)
        nsteps = size(positions, 3)
        length(t_s) == nsteps || error("Phase-space verification failed: t_s length mismatch for " * path_h5)
        length(t_norm) == nsteps || error("Phase-space verification failed: t_norm length mismatch for " * path_h5)
        length(t_gyroperiods) == nsteps || error("Phase-space verification failed: t_gyroperiods length mismatch for " * path_h5)
        RunnerMod.validate_time_axes(t_s, t_norm, t_gyroperiods; key_name="Phase-space cache time axes", require_uniform=true)
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
        t_gyroperiods = haskey(file, "t_gyroperiods") ? read(file["t_gyroperiods"]) : RunnerMod.t_gyroperiods_from_axes(t_s, t_norm)
        alive_fraction = read(file["alive_fraction"])
        nsteps = size(mu, 2)
        length(t_s) == nsteps || error("Mu-cache verification failed: t_s length mismatch for " * path_h5)
        length(t_norm) == nsteps || error("Mu-cache verification failed: t_norm length mismatch for " * path_h5)
        length(t_gyroperiods) == nsteps || error("Mu-cache verification failed: t_gyroperiods length mismatch for " * path_h5)
        RunnerMod.validate_time_axes(t_s, t_norm, t_gyroperiods; key_name="Mu-cache time axes", require_uniform=true)
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

function requested_default_injection(cfg)
    return RunnerMod.injection_mode(cfg) == :isotropic && RunnerMod.injection_position_mode(cfg) == :random
end

function read_h5_string(file, key::AbstractString)
    value = read(file[key])
    value isa AbstractArray && (value = first(value))
    return String(value)
end

function verify_cache_injection_metadata(path_h5::AbstractString, cfg)
    requested_default_injection(cfg) && return true
    h5open(path_h5, "r") do file
        required = ("injection_mode", "injection_mu0", "injection_position_mode", "injection_position", "injection_position_unit")
        missing = [key for key in required if !haskey(file, key)]
        isempty(missing) || error("Existing HDF5 lacks injection metadata required for non-default injection: " * join(missing, ", ") * ". Regenerate the cache/output for this injection configuration.")

        stored_mode = Symbol(read_h5_string(file, "injection_mode"))
        stored_pos_mode = Symbol(read_h5_string(file, "injection_position_mode"))
        stored_pos_unit = Symbol(read_h5_string(file, "injection_position_unit"))
        stored_mu0 = Float64(first(read(file["injection_mu0"])))
        stored_position = Float64.(read(file["injection_position"]))

        requested_mode = RunnerMod.injection_mode(cfg)
        requested_pos_mode = RunnerMod.injection_position_mode(cfg)
        requested_pos_unit = RunnerMod.injection_position_unit(cfg)
        requested_mu0 = Float64(get(cfg, :injection_mu0, 0.0))
        requested_position = Float64[Float64(v) for v in RunnerMod.injection_position_tuple(cfg, Float64)]

        stored_mode == requested_mode || error("Existing cache injection_mode=" * string(stored_mode) * " does not match requested " * string(requested_mode))
        stored_pos_mode == requested_pos_mode || error("Existing cache injection_position_mode=" * string(stored_pos_mode) * " does not match requested " * string(requested_pos_mode))
        stored_pos_unit == requested_pos_unit || error("Existing cache injection_position_unit=" * string(stored_pos_unit) * " does not match requested " * string(requested_pos_unit))
        isapprox(stored_mu0, requested_mu0; atol=1.0e-12, rtol=0.0) || error("Existing cache injection_mu0=" * string(stored_mu0) * " does not match requested " * string(requested_mu0))
        length(stored_position) == 3 || error("Existing cache injection_position metadata is malformed")
        all(isapprox.(stored_position, requested_position; atol=1.0e-12, rtol=0.0)) || error("Existing cache injection_position does not match requested injection_position")
    end
    return true
end

function verify_combined_outputs(path_h5::AbstractString, delta_png::AbstractString, heatmap_png::AbstractString, collapsed_png::AbstractString)
    verify_file_nonempty(path_h5)
    verify_file_nonempty(delta_png)
    verify_file_nonempty(heatmap_png)
    verify_file_nonempty(collapsed_png)

    h5open(path_h5, "r") do file
        delta_group = file["delta_mu2"]
        dmumu_group = file["dmumu"]
        haskey(file, "dpp") && error("Combined verification failed: D_pp output belongs in dpp_full.h5, not " * path_h5)
        haskey(file, "energy_snapshots") && error("Combined verification failed: energy snapshots belong in dpp_full.h5, not " * path_h5)
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

function resolve_energy_time_reference(cfg, reference_fields, energy_GeV, reference_h5::AbstractString)
    T = cfg[:precision]
    Bx, By, Bz = reference_fields[1], reference_fields[2], reference_fields[3]
    B0 = RunnerMod.reference_B0_T(Bx, By, Bz)
    gamma0 = RunnerMod.energy_to_gamma(energy_GeV, T)
    Omega0 = RunnerMod.reference_Omega0(B0, gamma0, RunnerMod.Q_E, RunnerMod.M_P)
    return RunnerMod.campaign_time_reference(
        B0,
        Omega0;
        source_mode="total",
        source_path=reference_h5,
        source_identity=reference_h5,
        field_subset=get(cfg, :field_subset, nothing),
    )
end

function expected_time_resolution(cfg, fields, energy_GeV, time_reference)
    T = cfg[:precision]
    Bx, By, Bz = fields[1], fields[2], fields[3]
    nx, ny, nz = size(Bx)
    x, y, z = RunnerMod.build_uniform_coords(cfg, nx, ny, nz, T)
    Omega0 = time_reference.Omega0_reference_s_inv
    v0 = RunnerMod.energy_to_speed(energy_GeV, T)
    trajectory_time = RunnerMod.resolve_trajectory_time_grid(cfg, Omega0, v0, RunnerMod.estimate_min_dx(x, y, z))
    save_time = RunnerMod.resolve_save_stride(cfg, trajectory_time)
    return trajectory_time, save_time, Float64(Omega0), Float64(time_reference.B0_reference_T)
end

function verify_cache_time_metadata(path_h5::AbstractString, cfg, fields, energy_GeV, time_reference)
    trajectory_time, save_time, expected_Omega0, expected_B0 = expected_time_resolution(cfg, fields, energy_GeV, time_reference)
    h5open(path_h5, "r") do file
        t_s = Float64.(read(file["t_s"]))
        t_norm = Float64.(read(file["t_norm"]))
        t_gyroperiods = haskey(file, "t_gyroperiods") ? Float64.(read(file["t_gyroperiods"])) : RunnerMod.t_gyroperiods_from_axes(t_s, t_norm)
        RunnerMod.validate_time_axes(t_s, t_norm, t_gyroperiods; key_name="Existing cache time axes", require_uniform=true)
        stored_Omega0 = h5_scalar_float_or(file, "Omega0", nothing)
        stored_B0 = h5_scalar_float_or(file, "B0_T", nothing)
        stored_dt = h5_scalar_float_or(file, "dt_s", nothing)
        stored_stride = h5_scalar_int_or(file, "trajectory_time_stride", save_time.trajectory_time_stride)
        stored_Omega0 === nothing || isapprox(stored_Omega0, expected_Omega0; rtol=1.0e-10, atol=0.0) || error("Existing cache Omega0 does not match requested time reference; regenerate " * path_h5)
        stored_B0 === nothing || isapprox(stored_B0, expected_B0; rtol=1.0e-10, atol=0.0) || error("Existing cache B0_T does not match requested time reference; regenerate " * path_h5)
        stored_reference_source = h5_scalar_string_or(file, "time_reference_source_path", "")
        isempty(stored_reference_source) || stored_reference_source == time_reference.time_reference_source_path || error("Existing cache time_reference_source_path does not match requested total-field reference; regenerate " * path_h5)
        stored_dt === nothing || isapprox(stored_dt, trajectory_time.dt_s; rtol=1.0e-10, atol=0.0) || error("Existing cache dt_s does not match requested integration resolution; regenerate " * path_h5)
        stored_stride == save_time.trajectory_time_stride || error("Existing cache trajectory_time_stride does not match requested save interval; regenerate " * path_h5)
        analysis_duration_gp = Float64(t_gyroperiods[end] - t_gyroperiods[1])
        expected_analysis_duration_gp = (length(t_gyroperiods) - 1) * save_time.actual_trajectory_save_interval_gyroperiods
        isapprox(analysis_duration_gp, expected_analysis_duration_gp; rtol=1.0e-10, atol=1.0e-12) || error("Existing cache analysis duration does not match requested uniform cache cadence; regenerate " * path_h5)
    end
    return true
end

function read_cache_time_summary(path_h5::AbstractString)
    h5open(path_h5, "r") do file
        t_s = Float64.(read(file["t_s"]))
        t_norm = Float64.(read(file["t_norm"]))
        t_gyroperiods = haskey(file, "t_gyroperiods") ? Float64.(read(file["t_gyroperiods"])) : RunnerMod.t_gyroperiods_from_axes(t_s, t_norm)
        return (
            t_s = t_s,
            t_norm = t_norm,
            t_gyroperiods = t_gyroperiods,
            Omega0 = h5_scalar_float_or(file, "Omega0", NaN),
            B0_T = h5_scalar_float_or(file, "B0_T", NaN),
            reference_gyroperiod_s = h5_scalar_float_or(file, "reference_gyroperiod_s", NaN),
            B0_definition = h5_scalar_string_or(file, "time_reference_B0_definition", RunnerMod.TIME_REFERENCE_B0_DEFINITION),
            requested_duration_gp = h5_scalar_float_or(file, "requested_trajectory_duration_gyroperiods", NaN),
            actual_duration_gp = h5_scalar_float_or(file, "actual_trajectory_duration_gyroperiods", Float64(t_gyroperiods[end] - t_gyroperiods[1])),
            requested_steps_per_gp = h5_scalar_float_or(file, "requested_integration_steps_per_gyroperiod", NaN),
            actual_steps_per_gp = h5_scalar_float_or(file, "actual_integration_steps_per_gyroperiod", NaN),
            timestep_limited_by = h5_scalar_string_or(file, "timestep_limited_by", "unknown"),
            requested_save_gp = h5_scalar_float_or(file, "requested_trajectory_save_interval_gyroperiods", NaN),
            actual_save_gp = h5_scalar_float_or(file, "actual_trajectory_save_interval_gyroperiods", NaN),
        )
    end
end

function lag_summary_for(cfg, t_gyroperiods)
    lag_grid = CombinedFullMod.resolve_lag_grid(cfg, t_gyroperiods)
    return (requested_min=first(lag_grid.requested_tau_gyroperiods), requested_max=last(lag_grid.requested_tau_gyroperiods), actual_min=first(lag_grid.tau_gyroperiods), actual_max=last(lag_grid.tau_gyroperiods), count=length(lag_grid.lag_steps))
end

function print_resolved_time_summary(cache_h5::AbstractString, dmumu_cfg, dpp_cfg)
    summary = read_cache_time_summary(cache_h5)
    println("Time reference:")
    println("  Omega0                         = ", summary.Omega0, " s^-1")
    println("  reference gyroperiod           = ", summary.reference_gyroperiod_s, " s")
    println("  B0 definition                  = ", summary.B0_definition)
    println("Trajectory:")
    println("  requested duration             = ", summary.requested_duration_gp, " reference gyroperiods")
    println("  actual duration                = ", summary.actual_duration_gp, " reference gyroperiods")
    println("  requested integration res.     = ", summary.requested_steps_per_gp, " steps/reference gyroperiod")
    println("  actual integration res.        = ", summary.actual_steps_per_gp, " steps/reference gyroperiod")
    println("  timestep limiter               = ", summary.timestep_limited_by)
    println("  requested save interval        = ", summary.requested_save_gp, " reference gyroperiods")
    println("  actual save interval           = ", summary.actual_save_gp, " reference gyroperiods")
    println("  saved samples                  = ", length(summary.t_gyroperiods))
    if dmumu_cfg !== nothing
        dmumu_lags = lag_summary_for(dmumu_cfg, summary.t_gyroperiods)
        println("D_mumu lag grid:")
        println("  requested range                = ", dmumu_lags.requested_min, " to ", dmumu_lags.requested_max, " reference gyroperiods")
        println("  actual range                   = ", dmumu_lags.actual_min, " to ", dmumu_lags.actual_max, " reference gyroperiods")
        println("  unique lag count               = ", dmumu_lags.count)
    end
    if dpp_cfg !== nothing
        dpp_lags = lag_summary_for(dpp_cfg, summary.t_gyroperiods)
        println("D_pp lag grid:")
        println("  requested range                = ", dpp_lags.requested_min, " to ", dpp_lags.requested_max, " reference gyroperiods")
        println("  actual range                   = ", dpp_lags.actual_min, " to ", dpp_lags.actual_max, " reference gyroperiods")
        println("  unique lag count               = ", dpp_lags.count)
    end
    return nothing
end

function verify_dpp_outputs(path_h5::AbstractString, path_png::AbstractString)
    verify_file_nonempty(path_h5)
    verify_file_nonempty(path_png)
    h5open(path_h5, "r") do file
        dpp_group = file["dpp"]
        energy_group = file["energy_snapshots"]
        dpp_tau_norm = read(dpp_group["tau_norm"])
        dpp_centered = read(dpp_group["D_pp_centered_norm"])
        energies = read(energy_group["energy_GeV"])
        length(dpp_tau_norm) == length(dpp_centered) || error("D_pp verification failed: tau/curve mismatch for " * path_h5)
        any(isfinite, dpp_centered) || error("D_pp verification failed: no finite D_pp values in " * path_h5)
        size(energies, 1) > 0 || error("D_pp verification failed: no energy snapshots in " * path_h5)
        any(isfinite, energies) || error("D_pp verification failed: no finite snapshot energies in " * path_h5)
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

function h5_scalar_float_or(file, dataset_name::AbstractString, default)
    value = h5_scalar_value_or(file, dataset_name, default)
    value === nothing && return nothing
    value isa Real && return Float64(value)
    return parse(Float64, string(value))
end

function optional_float_matches(stored, requested; atol=1.0e-10, rtol=1.0e-10)
    requested === nothing && return stored === nothing
    stored === nothing && return false
    return isapprox(Float64(stored), Float64(requested); atol=atol, rtol=rtol)
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
        if !(haskey(file, "delta_mu2") && haskey(file, "dmumu"))
            return false
        end
        stored_start_mode = CombinedFullMod.parse_dmumu_start_mode(h5_scalar_string_or(file, "dmumu_start_mode", "sliding"))
        stored_mu_bin_abs = h5_scalar_bool_or(file, "mu_bin_abs", false)
        stored_lag_mode = CombinedFullMod.parse_lag_mode(h5_scalar_string_or(file, "lag_mode", "uniform_samples"))
        stored_lag_sample_count = h5_scalar_int_or(file, "n_lag_samples", Int(cfg[:n_lag_samples]))
        stored_requested_n_lag_samples = h5_scalar_int_or(file, "requested_n_lag_samples", stored_lag_sample_count)
        stored_min_lag_steps = h5_scalar_int_or(file, "min_lag_steps", 1)
        stored_lag_step_stride = h5_scalar_int_or(file, "lag_step_stride", 1)
        stored_max_lag_steps = h5_scalar_int_or(file, "max_lag_steps", -1)
        stored_lag_min_gp = h5_scalar_float_or(file, "lag_min_gyroperiods", nothing)
        stored_lag_max_gp = h5_scalar_float_or(file, "lag_max_gyroperiods", nothing)
        stored_lag_stride_gp = h5_scalar_float_or(file, "lag_stride_gyroperiods", nothing)
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
               optional_float_matches(stored_lag_min_gp, get(cfg, :lag_min_gyroperiods, nothing)) &&
               optional_float_matches(stored_lag_max_gp, get(cfg, :lag_max_gyroperiods, nothing)) &&
               optional_float_matches(stored_lag_stride_gp, get(cfg, :lag_stride_gyroperiods, nothing)) &&
               stored_n_mu_bins == Int(cfg[:n_mu_bins]) &&
               length(stored_mu_edges) == Int(cfg[:n_mu_bins]) + 1 &&
               isapprox(first(stored_mu_edges), Float64(cfg[:mu_min]); atol=0.0, rtol=0.0) &&
               isapprox(last(stored_mu_edges), Float64(cfg[:mu_max]); atol=0.0, rtol=0.0)
    end
end

function dpp_outputs_match_requested(path_h5::AbstractString, cfg)
    h5open(path_h5, "r") do file
        haskey(file, "dpp") && haskey(file, "energy_snapshots") || return false
        stored_lag_mode = DppFullMod.parse_lag_mode(h5_scalar_string_or(file, "lag_mode", "uniform_samples"))
        stored_lag_sample_count = h5_scalar_int_or(file, "n_lag_samples", Int(cfg[:n_lag_samples]))
        stored_requested_n_lag_samples = h5_scalar_int_or(file, "requested_n_lag_samples", stored_lag_sample_count)
        stored_min_lag_steps = h5_scalar_int_or(file, "min_lag_steps", 1)
        stored_lag_step_stride = h5_scalar_int_or(file, "lag_step_stride", 1)
        stored_max_lag_steps = h5_scalar_int_or(file, "max_lag_steps", -1)
        stored_lag_min_gp = h5_scalar_float_or(file, "lag_min_gyroperiods", nothing)
        stored_lag_max_gp = h5_scalar_float_or(file, "lag_max_gyroperiods", nothing)
        stored_lag_stride_gp = h5_scalar_float_or(file, "lag_stride_gyroperiods", nothing)
        requested_max_lag_steps = cfg[:max_lag_steps] === nothing ? -1 : Int(cfg[:max_lag_steps])
        return stored_lag_mode == cfg[:lag_mode] &&
               stored_requested_n_lag_samples == Int(cfg[:n_lag_samples]) &&
               stored_min_lag_steps == Int(get(cfg, :min_lag_steps, 1)) &&
               stored_lag_step_stride == Int(cfg[:lag_step_stride]) &&
               stored_max_lag_steps == requested_max_lag_steps &&
               optional_float_matches(stored_lag_min_gp, get(cfg, :lag_min_gyroperiods, nothing)) &&
               optional_float_matches(stored_lag_max_gp, get(cfg, :lag_max_gyroperiods, nothing)) &&
               optional_float_matches(stored_lag_stride_gp, get(cfg, :lag_stride_gyroperiods, nothing))
    end
end

function dmumu_outputs_complete(paths, cfg)
    try
        verify_combined_outputs(paths.combined_h5, paths.delta_png, paths.dmumu_heatmap_png, paths.dmumu_collapsed_png)
        combined_outputs_match_requested(paths.combined_h5, cfg) || return false
        return true
    catch
        return false
    end
end

function dpp_outputs_complete(paths, cfg)
    try
        verify_dpp_outputs(paths.dpp_h5, paths.dpp_png)
        dpp_outputs_match_requested(paths.dpp_h5, cfg) || return false
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

function summary_row(
    energy_GeV,
    status::AbstractString,
    deleted::Bool,
    cache_path::AbstractString,
    dmumu_h5::AbstractString,
    dpp_h5::AbstractString,
    dmumu_status::AbstractString,
    dpp_status::AbstractString,
    dmumu_note::AbstractString,
    dpp_note::AbstractString,
    note::AbstractString,
)
    return (
        energy_GeV = Float64(energy_GeV),
        status = status,
        deleted = deleted,
        cache_path = cache_path,
        dmumu_h5 = dmumu_h5,
        dpp_h5 = dpp_h5,
        dmumu_status = dmumu_status,
        dpp_status = dpp_status,
        dmumu_note = dmumu_note,
        dpp_note = dpp_note,
        note = note,
    )
end

function write_summary(path::AbstractString, rows)
    ensure_parent(path)
    open(path, "w") do io
        println(io, "energy_GeV\tstatus\tcache_deleted\tcache_path\tdmumu_h5\tdpp_h5\tdmumu_status\tdpp_status\tdmumu_note\tdpp_note\tnote")
        for row in rows
            println(
                io,
                join((
                    @sprintf("%.6f", row.energy_GeV),
                    row.status,
                    row.deleted ? "true" : "false",
                    row.cache_path,
                    row.dmumu_h5,
                    row.dpp_h5,
                    row.dmumu_status,
                    row.dpp_status,
                    replace(row.dmumu_note, '\n' => ' '),
                    replace(row.dpp_note, '\n' => ' '),
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
    dpp_h5 = joinpath(science_dir, "dpp_full.h5")
    delta_png = joinpath(science_dir, "delta_mu2_curve_full.png")
    dmumu_heatmap_png = joinpath(science_dir, "dmumu_mu_tau_full.png")
    dmumu_collapsed_png = joinpath(science_dir, "dmumu_tau_average_full.png")
    dpp_png = joinpath(science_dir, "dpp_tau_curve_full.png")
    return (
        cache_dir = cache_dir,
        science_dir = science_dir,
        cache_h5 = cache_h5,
        combined_h5 = combined_h5,
        dpp_h5 = dpp_h5,
        delta_png = delta_png,
        dmumu_heatmap_png = dmumu_heatmap_png,
        dmumu_collapsed_png = dmumu_collapsed_png,
        dpp_png = dpp_png,
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

function finalize_mu_cache_writer!(writer, cfg, energy_GeV, t_norm_save, t_s_save, t_gyroperiods_save, alive_fraction_save, trajectory_time, save_time, time_reference)
    file = writer.file
    file["t_norm"] = Float64.(t_norm_save)
    file["t_s"] = Float64.(t_s_save)
    file["t_gyroperiods"] = Float64.(t_gyroperiods_save)
    file["alive_fraction"] = Float64.(alive_fraction_save)
    file["energy_GeV"] = Float64[energy_GeV]
    file["n_particles"] = Int[cfg[:n_particles]]
    RunnerMod.write_time_reference_metadata!(file, time_reference)
    RunnerMod.write_trajectory_time_metadata!(file, trajectory_time, save_time)
    file["analysis_cache_duration_gyroperiods"] = Float64[t_gyroperiods_save[end] - t_gyroperiods_save[1]]
    file["integration_step_count"] = Int[trajectory_time.n_integration_steps]
    file["analysis_sample_count"] = Int[length(t_gyroperiods_save)]
    file["exact_final_state_stored"] = false
    file["trajectory_mode"] = string(get(cfg, :trajectory_mode, "unknown"))
    file["trajectory_field_source_path"] = string(get(cfg, :trajectory_field_source_path, cfg[:file]))
    file["trajectory_field_source_identity"] = string(get(cfg, :trajectory_field_source_identity, get(cfg, :trajectory_field_source_path, cfg[:file])))
    file["boundary_mode"] = string(cfg[:boundary])
    file["cache_mode"] = "mu"
    file["cache_output_precision"] = string(writer.outtype)
    file["mu_unit"] = "dimensionless"
    RunnerMod.write_injection_metadata!(file, cfg)
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

function run_mu_cache_ensemble(cfg, fields, output_h5::AbstractString; energy_GeV, time_reference=nothing)
    T = cfg[:precision]
    Bx, By, Bz, vx, vy, vz = fields
    nx, ny, nz = size(Bx)
    x, y, z = RunnerMod.build_uniform_coords(cfg, nx, ny, nz, T)

    boundary = cfg[:boundary]
    if !(boundary == :open || boundary == :periodic)
        error("Unknown boundary mode: " * string(boundary) * ". Use :open or :periodic.")
    end
    periodic_boundary = boundary == :periodic

    cache_out_type = cfg[:mu_cache_output_precision]

    gamma0 = RunnerMod.energy_to_gamma(energy_GeV, T)
    if time_reference === nothing
        B0 = RunnerMod.reference_B0_T(Bx, By, Bz)
        Omega0_value = RunnerMod.reference_Omega0(B0, gamma0, RunnerMod.Q_E, RunnerMod.M_P)
        time_reference = RunnerMod.campaign_time_reference(B0, Omega0_value; source_mode="trajectory", source_path=String(cfg[:file]), source_identity=String(cfg[:file]), field_subset=get(cfg, :field_subset, nothing))
    end
    Omega0 = T(time_reference.Omega0_reference_s_inv)
    v0 = RunnerMod.energy_to_speed(energy_GeV, T)
    trajectory_time = RunnerMod.resolve_trajectory_time_grid(cfg, Omega0, v0, RunnerMod.estimate_min_dx(x, y, z))
    save_time = RunnerMod.resolve_save_stride(cfg, trajectory_time)
    trajectory_stride = save_time.trajectory_time_stride

    dt = T(trajectory_time.dt_s)
    nsteps = trajectory_time.n_integration_steps
    t_s = dt .* collect(T, 0:(nsteps - 1))
    t_norm = t_s .* Omega0
    t_gyroperiods = t_norm ./ T(2 * pi)
    save_indices = RunnerMod.sampled_step_indices(nsteps, trajectory_stride)
    nsave = length(save_indices)

    x1, x2, x3, p1, p2, p3 = RunnerMod.init_particles(cfg, x, y, z, gamma0, v0, T, Bx, By, Bz)
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
    println("  duration [Tg0] = ", trajectory_time.actual_trajectory_duration_gyroperiods)
    println("  dt [Tg0]       = ", trajectory_time.dt_gyroperiods)
    if trajectory_time.timestep_limited_by == "cfl"
        println("  CFL reduced timestep; actual steps/reference gyroperiod = ", trajectory_time.actual_integration_steps_per_gyroperiod)
    end
    if save_time.save_interval_source == :gyroperiods && abs(save_time.actual_trajectory_save_interval_gyroperiods - save_time.requested_trajectory_save_interval_gyroperiods) > 1.0e-6 * max(1.0, abs(save_time.requested_trajectory_save_interval_gyroperiods))
        @warn "Requested trajectory save interval was rounded to an integer integration-step stride." requested_gyroperiods=save_time.requested_trajectory_save_interval_gyroperiods actual_gyroperiods=save_time.actual_trajectory_save_interval_gyroperiods trajectory_time_stride=trajectory_stride
    end
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
            t_gyroperiods[save_indices],
            alive_fraction[save_indices],
            trajectory_time,
            save_time,
            time_reference,
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
        trajectory_time = trajectory_time,
        save_time = save_time,
        B0 = Float64(time_reference.B0_reference_T),
        time_reference = time_reference,
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
        h5open(cache_h5, "r") do cache_file
            for key in (
                "energy_GeV",
                "dt_s",
                "dt_tOmega0",
                "dt_gyroperiods",
                "Omega0",
                "B0_T",
                "Omega0_reference_s_inv",
                "B0_reference_T",
                "reference_gyroperiod_s",
                "time_reference_name",
                "time_reference_mode",
                "time_reference_definition",
                "time_reference_B0_definition",
                "time_reference_energy_definition",
                "time_reference_source_mode",
                "time_reference_source_path",
                "time_reference_source_identity",
                "time_reference_field_subset",
                "requested_trajectory_duration_gyroperiods",
                "actual_trajectory_duration_gyroperiods",
                "actual_trajectory_duration_tOmega0",
                "actual_trajectory_duration_s",
                "requested_integration_steps_per_gyroperiod",
                "actual_integration_steps_per_gyroperiod",
                "timestep_limited_by",
                "requested_trajectory_save_interval_gyroperiods",
                "actual_trajectory_save_interval_gyroperiods",
                "n_particles",
                "trajectory_time_stride",
                "analysis_cache_duration_gyroperiods",
                "integration_step_count",
                "analysis_sample_count",
                "exact_final_state_stored",
                "trajectory_mode",
                "trajectory_field_source_path",
                "trajectory_field_source_identity",
                "boundary_mode",
                "trajectory_output_precision",
                "cache_output_precision",
                "position_unit",
                "momentum_unit",
                "mu_unit",
                "injection_mode",
                "injection_mu0",
                "injection_position_mode",
                "injection_position",
                "injection_position_unit",
            )
                haskey(cache_file, key) && !haskey(file, key) && (file[key] = read(cache_file[key]))
            end
        end
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
        t_gyroperiods = haskey(cache_file, "t_gyroperiods") ? Float64.(read(cache_file["t_gyroperiods"])) : CombinedFullMod.t_gyroperiods_from_axes(t_s, t_norm)
        nsteps = size(mu_dataset, 2)
        CombinedFullMod.validate_time_axes(t_s, t_norm, t_gyroperiods; key_name="Mu-cache time axes", require_uniform=true)
        total_particles = size(mu_dataset, 1)

        particle_indices = CombinedFullMod.build_particle_indices(total_particles, cfg)
        first_particle = first(particle_indices)
        last_particle = last(particle_indices)
        selected_particle_count = length(particle_indices)
        selected_particle_count > 0 || error("No particles selected.")

        lag_grid = CombinedFullMod.resolve_lag_grid(cfg, t_gyroperiods)
        if lag_grid.duplicate_lag_mapping_count > 0 && lag_grid.duplicate_lag_mapping_count / lag_grid.requested_lag_count > 0.10
            @warn "Requested D_mumu lag grid collapsed substantially after mapping to cached-step offsets." requested=lag_grid.requested_lag_count unique=lag_grid.unique_lag_count duplicates=lag_grid.duplicate_lag_mapping_count
        end
        lag_steps = lag_grid.lag_steps
        tau_s = [Float64(t_s[lag_step + 1] - t_s[1]) for lag_step in lag_steps]
        tau_norm = Float64.(lag_grid.tau_norm)
        tau_gyroperiods = Float64.(lag_grid.tau_gyroperiods)
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
            collapsed_centered_norm_count_weighted;
            lag_grid=lag_grid,
        )
        append_cache_metadata!(cfg[:output_h5], :mu, cfg[:cache_h5])
        println("Saved combined HDF5 to ", cfg[:output_h5])

        CombinedFullMod.plot_delta_mu2(delta_df, cfg[:output_delta_png]; use_usetex=cfg[:use_usetex])
        println("Saved delta_mu2 plot to ", cfg[:output_delta_png])

        CombinedFullMod.plot_dmumu_heatmap_full(
            cfg[:output_heatmap_png],
            mu_edges,
            tau_gyroperiods,
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

function throwable_note(error_instance, backtrace_value)
    return sprint(showerror, error_instance, backtrace_value)
end

function requested_status(compute_product::Bool, complete::Bool)
    compute_product || return "disabled"
    complete && return "skipped_existing"
    return "pending"
end

function product_success(status::AbstractString)
    return status in ("disabled", "skipped_existing", "ok")
end

function run_dmumu_product!(cfg, paths)
    if cfg[:cache_mode] == :phase_space
        CombinedFullMod.run_combined_full(cfg)
        append_cache_metadata!(paths.combined_h5, :phase_space, paths.cache_h5)
    else
        run_combined_from_mu_cache(cfg)
    end
    verify_combined_outputs(paths.combined_h5, paths.delta_png, paths.dmumu_heatmap_png, paths.dmumu_collapsed_png)
    return nothing
end

function run_dpp_product!(cfg, paths)
    cfg[:cache_mode] == :phase_space || error("D_pp requires phase-space cache mode.")
    DppFullMod.run_dpp_full(cfg)
    verify_dpp_outputs(paths.dpp_h5, paths.dpp_png)
    return nothing
end

function overall_product_status(dmumu_status::AbstractString, dpp_status::AbstractString)
    if dmumu_status == "error" || dpp_status == "error"
        return "error"
    elseif dmumu_status == "ok" || dpp_status == "ok"
        return "ok"
    elseif dmumu_status == "skipped_existing" || dpp_status == "skipped_existing"
        return "skipped_existing_outputs"
    end
    return "cache_only"
end

function run_energy_pipeline!(cfg, fields, reference_fields, base_runner_cfg, energy_GeV)
    paths = build_energy_paths(cfg, energy_GeV)
    mkpath(paths.cache_dir)
    mkpath(paths.science_dir)
    requested_dmumu_cfg = merge_cfg(CombinedFullMod.COMBINED_FULL_CFG, cfg[:dmumu_overrides])
    requested_dpp_cfg = merge_cfg(DppFullMod.DPP_FULL_CFG, cfg[:dpp_overrides])

    println()
    println("=== Energy ", energy_GeV, " GeV ===")
    time_reference = resolve_energy_time_reference(base_runner_cfg, reference_fields, energy_GeV, cfg[:reference_h5])
    println("Energy:")
    println("  energy                         = ", energy_GeV, " GeV")
    println("  Omega0_reference               = ", time_reference.Omega0_reference_s_inv, " s^-1")
    println("  reference gyroperiod           = ", time_reference.reference_gyroperiod_s, " s")
    compute_analysis = cfg[:compute_dmumu] || cfg[:compute_dpp]

    if !compute_analysis && cfg[:skip_completed_outputs] && isfile(paths.cache_h5)
        verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
        verify_cache_injection_metadata(paths.cache_h5, base_runner_cfg)
        verify_cache_time_metadata(paths.cache_h5, base_runner_cfg, fields, energy_GeV, time_reference)
        println("Cache already verified; transport analysis disabled for energy ", energy_GeV, " GeV")
        return summary_row(
            energy_GeV,
            "skipped_existing_cache",
            false,
            paths.cache_h5,
            "",
            "",
            "disabled",
            "disabled",
            "",
            "",
            "verified existing cache; transport analysis disabled",
        )
    end

    dmumu_complete = !cfg[:compute_dmumu] || (cfg[:skip_completed_outputs] && dmumu_outputs_complete(paths, requested_dmumu_cfg))
    dpp_complete = !cfg[:compute_dpp] || (cfg[:skip_completed_outputs] && dpp_outputs_complete(paths, requested_dpp_cfg))
    dmumu_status = requested_status(cfg[:compute_dmumu], dmumu_complete)
    dpp_status = requested_status(cfg[:compute_dpp], dpp_complete)
    dmumu_note = dmumu_status == "skipped_existing" ? "verified existing D_mumu outputs" : ""
    dpp_note = dpp_status == "skipped_existing" ? "verified existing D_pp outputs" : ""

    if cfg[:skip_completed_outputs] && dmumu_complete && dpp_complete
        metadata_path = isfile(paths.cache_h5) ? paths.cache_h5 : (cfg[:compute_dmumu] ? paths.combined_h5 : paths.dpp_h5)
        verify_cache_injection_metadata(metadata_path, base_runner_cfg)
        deleted = false
        if isfile(paths.cache_h5)
            verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
            verify_cache_injection_metadata(paths.cache_h5, base_runner_cfg)
            verify_cache_time_metadata(paths.cache_h5, base_runner_cfg, fields, energy_GeV, time_reference)
            deleted = delete_cache_if_requested(paths.cache_h5, cfg)
        end
        println("Requested outputs already verified; skipping energy ", energy_GeV, " GeV")
        return summary_row(
            energy_GeV,
            "skipped_existing_outputs",
            deleted,
            paths.cache_h5,
            paths.combined_h5,
            paths.dpp_h5,
            dmumu_status,
            dpp_status,
            dmumu_note,
            dpp_note,
            "verified existing requested outputs",
        )
    end

    if cfg[:reuse_existing_cache] && isfile(paths.cache_h5)
        println("Reusing existing cache file ", paths.cache_h5)
        verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
        verify_cache_injection_metadata(paths.cache_h5, base_runner_cfg)
        verify_cache_time_metadata(paths.cache_h5, base_runner_cfg, fields, energy_GeV, time_reference)
    else
        if cfg[:cache_mode] == :phase_space
            runner_cfg = merge_cfg(base_runner_cfg, Dict{Symbol, Any}(
                :output_dir => paths.cache_dir,
                :energies => [energy_GeV],
            ))
            runner_result = RunnerMod.run_gpu_ensemble(runner_cfg, fields; energy_GeV=energy_GeV, time_reference=time_reference)
            verify_phase_space_cache_h5(runner_result.phase_space_path)
        elseif cfg[:cache_mode] == :mu
            run_mu_cache_ensemble(base_runner_cfg, fields, paths.cache_h5; energy_GeV=energy_GeV, time_reference=time_reference)
            verify_mu_cache_h5(paths.cache_h5)
        else
            error("Unknown cache mode: " * string(cfg[:cache_mode]))
        end
    end

    if !compute_analysis
        verify_cache_h5(paths.cache_h5, cfg[:cache_mode])
        verify_cache_injection_metadata(paths.cache_h5, base_runner_cfg)
        verify_cache_time_metadata(paths.cache_h5, base_runner_cfg, fields, energy_GeV, time_reference)
        println("Transport analysis disabled; keeping verified cache file ", paths.cache_h5)
        return summary_row(
            energy_GeV,
            "cache_only",
            false,
            paths.cache_h5,
            "",
            "",
            "disabled",
            "disabled",
            "",
            "",
            "verified cache; transport analysis disabled",
        )
    end

    dmumu_cfg = merge_cfg(
        requested_dmumu_cfg,
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

    dpp_cfg = merge_cfg(
        requested_dpp_cfg,
        Dict{Symbol, Any}(
            :trajectory_h5 => paths.cache_h5,
            :cache_h5 => paths.cache_h5,
            :cache_mode => cfg[:cache_mode],
            :output_dir => paths.science_dir,
            :output_h5 => paths.dpp_h5,
            :output_dpp_png => paths.dpp_png,
        ),
    )

    print_resolved_time_summary(
        paths.cache_h5,
        cfg[:compute_dmumu] ? dmumu_cfg : nothing,
        cfg[:compute_dpp] ? dpp_cfg : nothing,
    )

    if cfg[:compute_dmumu] && !dmumu_complete
        try
            run_dmumu_product!(dmumu_cfg, paths)
            dmumu_status = "ok"
            dmumu_note = "verified D_mumu outputs"
        catch error_instance
            dmumu_status = "error"
            dmumu_note = throwable_note(error_instance, catch_backtrace())
            println("D_mumu failed for energy ", energy_GeV, " GeV; continuing with other requested products.")
            println(dmumu_note)
        end
    elseif !cfg[:compute_dmumu]
        rm(paths.delta_png; force=true)
        rm(paths.dmumu_heatmap_png; force=true)
        rm(paths.dmumu_collapsed_png; force=true)
    end

    if cfg[:compute_dpp] && !dpp_complete
        try
            run_dpp_product!(dpp_cfg, paths)
            dpp_status = "ok"
            dpp_note = "verified D_pp outputs"
        catch error_instance
            dpp_status = "error"
            dpp_note = throwable_note(error_instance, catch_backtrace())
            println("D_pp failed for energy ", energy_GeV, " GeV; continuing with other requested products.")
            println(dpp_note)
        end
    end

    all_requested_succeeded = product_success(dmumu_status) && product_success(dpp_status)
    deleted = false
    if all_requested_succeeded
        deleted = delete_cache_if_requested(paths.cache_h5, cfg)
    else
        println("Keeping cache file ", paths.cache_h5, " because at least one requested product failed.")
    end

    overall_status = overall_product_status(dmumu_status, dpp_status)
    overall_note = all_requested_succeeded ? "verified requested outputs" : "one or more requested products failed"
    return summary_row(
        energy_GeV,
        overall_status,
        deleted,
        paths.cache_h5,
        paths.combined_h5,
        paths.dpp_h5,
        dmumu_status,
        dpp_status,
        dmumu_note,
        dpp_note,
        overall_note,
    )
end

function run_energy_distribution_plots(campaign_root::AbstractString)
    script_path = joinpath(PIPELINE_ROOT, "scripts", "plot_energy_distribution.py")
    isfile(script_path) || error("Energy distribution plotter not found: " * script_path)
    python = get(ENV, "PYTHON", "python")
    println("Generating campaign energy distribution plots with ", script_path)
    run(`$python $script_path $campaign_root`)
    return nothing
end

function run_campaign(cfg)
    mkpath(cfg[:campaign_root])
    base_runner_cfg = merge_cfg(
        RunnerMod.CFG,
        merge_cfg(
            cfg[:trajectory_overrides],
            Dict{Symbol, Any}(
                :file => cfg[:turbulence_h5],
                :trajectory_mode => cfg[:mode_name],
                :trajectory_field_source_path => cfg[:turbulence_h5],
                :trajectory_field_source_identity => cfg[:turbulence_h5],
                :mu_cache_output_precision => cfg[:mu_cache_output_precision],
            ),
        ),
    )

    println("Campaign ", cfg[:campaign_tag])
    println("  campaign root            = ", cfg[:campaign_root])
    println("  turbulence H5            = ", cfg[:turbulence_h5])
    println("  total reference H5       = ", cfg[:reference_h5])
    println("  cache mode               = ", cfg[:cache_mode])
    println("Shared time reference:")
    println("  source mode              = total")
    println("  source file              = ", cfg[:reference_h5])
    println("  definition               = mean |B_total| on loaded trajectory grid")
    println("Trajectory mode:")
    println("  mode                     = ", cfg[:mode_name])
    println("  trajectory field file    = ", cfg[:turbulence_h5])
    println("  time reference source    = total")
    println("Loading turbulence fields once for trajectory generation")
    fields = RunnerMod.load_static_fields(base_runner_cfg, base_runner_cfg[:precision])
    reference_runner_cfg = merge_cfg(base_runner_cfg, Dict{Symbol, Any}(:file => cfg[:reference_h5]))
    println("Loading total-field reference once for time normalization")
    reference_fields = RunnerMod.load_static_fields(reference_runner_cfg, reference_runner_cfg[:precision])

    summary_rows = Any[]
    summary_path = joinpath(cfg[:campaign_root], "campaign_summary.tsv")
    for energy_GeV in cfg[:energies]
        try
            row = run_energy_pipeline!(cfg, fields, reference_fields, base_runner_cfg, energy_GeV)
            push!(summary_rows, row)
            write_summary(summary_path, summary_rows)
            if cfg[:stop_on_error] && row.status == "error"
                println("Stopping campaign after requested product failure for energy ", energy_GeV, " GeV. See ", summary_path)
                return summary_rows
            end
        catch error_instance
            note = sprint(showerror, error_instance, catch_backtrace())
            paths = build_energy_paths(cfg, energy_GeV)
            push!(summary_rows, summary_row(
                energy_GeV,
                "error",
                false,
                paths.cache_h5,
                paths.combined_h5,
                paths.dpp_h5,
                cfg[:compute_dmumu] ? "error" : "disabled",
                cfg[:compute_dpp] ? "error" : "disabled",
                cfg[:compute_dmumu] ? note : "",
                cfg[:compute_dpp] ? note : "",
                note,
            ))
            write_summary(summary_path, summary_rows)
            if cfg[:stop_on_error]
                rethrow()
            end
        end
    end

    write_summary(summary_path, summary_rows)
    if cfg[:compute_dpp] && any(row -> row.dpp_status in ("ok", "skipped_existing"), summary_rows)
        run_energy_distribution_plots(cfg[:campaign_root])
    end
    println()
    println("Saved campaign summary to ", summary_path)
    return summary_rows
end

function materialize_campaign_cfg(cfg, campaign_spec)
    campaign_cfg = Dict{Symbol, Any}(
        :campaign_tag => campaign_spec[:campaign_tag],
        :campaign_root => campaign_spec[:campaign_root],
        :turbulence_h5 => campaign_spec[:turbulence_h5],
        :reference_h5 => campaign_spec[:reference_h5],
        :mode_name => campaign_spec[:mode_name],
        :cache_mode => cfg[:cache_mode],
        :compute_dmumu => cfg[:compute_dmumu],
        :compute_dpp => cfg[:compute_dpp],
        :energies => get(campaign_spec, :energies, cfg[:energies]),
        :delete_cache_on_success => cfg[:delete_cache_on_success],
        :reuse_existing_cache => cfg[:reuse_existing_cache],
        :skip_completed_outputs => cfg[:skip_completed_outputs],
        :stop_on_error => cfg[:stop_on_error],
        :run_all_particles_dmumu => cfg[:run_all_particles_dmumu],
        :mu_cache_output_precision => cfg[:mu_cache_output_precision],
        :trajectory_overrides => shallow_copy_dict(cfg[:trajectory_overrides]),
        :dmumu_overrides => shallow_copy_dict(cfg[:dmumu_overrides]),
        :dpp_overrides => shallow_copy_dict(cfg[:dpp_overrides]),
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

function config_tuple3_floats(value, key_name::AbstractString)
    values = if value isa AbstractString
        selected = split_csv_selector(value; flag_name=key_name)
        selected === nothing && error(key_name * " must contain three numeric values.")
        [parse(Float64, item) for item in selected]
    elseif value isa AbstractVector || value isa Tuple
        [config_float(item, key_name) for item in value]
    else
        error(key_name * " must be a string or array of three numeric values.")
    end
    length(values) == 3 || error(key_name * " must contain exactly three numeric values.")
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

function warn_legacy_config_keys(keys, replacement::AbstractString)
    isempty(keys) && return nothing
    @warn "Legacy time configuration key(s) are still supported but deprecated; use $(replacement)." keys=join(keys, ", ")
    return nothing
end

function reject_duplicate_time_controls(section, old_keys, new_keys, section_name::AbstractString, replacement::AbstractString)
    used_old = [key for key in old_keys if haskey(section, key)]
    used_new = [key for key in new_keys if haskey(section, key)]
    if !isempty(used_old) && !isempty(used_new)
        error(section_name * " cannot mix legacy key(s) " * join(used_old, ", ") * " with preferred key(s) " * join(used_new, ", ") * ". Use " * replacement * ".")
    end
    warn_legacy_config_keys(used_old, replacement)
    return nothing
end

function record_time_keys!(cfg, family::Symbol, source::Symbol, keys)
    haskey(cfg, :explicit_time_keys) || (cfg[:explicit_time_keys] = Dict{Symbol, Dict{Symbol, Set{String}}}())
    family_map = get!(cfg[:explicit_time_keys], family, Dict{Symbol, Set{String}}())
    source_set = get!(family_map, source, Set{String}())
    union!(source_set, String.(keys))
    return nothing
end

function reject_recorded_time_conflicts(cfg)
    haskey(cfg, :explicit_time_keys) || return nothing
    pairs = Dict(
        :trajectory_duration => (["tOmega0_max"], ["trajectory_duration_gyroperiods"]),
        :integration_resolution => (["eta"], ["integration_steps_per_gyroperiod"]),
        :trajectory_save_interval => (["trajectory_time_stride"], ["trajectory_save_interval_gyroperiods"]),
        :dmumu_lag_min => (["min_lag_steps"], ["lag_min_gyroperiods"]),
        :dmumu_lag_max => (["max_lag_steps"], ["lag_max_gyroperiods"]),
        :dmumu_lag_stride => (["lag_step_stride"], ["lag_stride_gyroperiods"]),
        :dpp_lag_min => (["min_lag_steps"], ["lag_min_gyroperiods"]),
        :dpp_lag_max => (["max_lag_steps"], ["lag_max_gyroperiods"]),
        :dpp_lag_stride => (["lag_step_stride"], ["lag_stride_gyroperiods"]),
    )
    for (family, (legacy, preferred)) in pairs
        haskey(cfg[:explicit_time_keys], family) || continue
        all_keys = String[]
        sources = String[]
        for (source, keys) in cfg[:explicit_time_keys][family]
            append!(all_keys, collect(keys))
            append!(sources, [string(source) * ":" * key for key in keys])
        end
        used_old = intersect(all_keys, legacy)
        used_new = intersect(all_keys, preferred)
        if !isempty(used_old) && !isempty(used_new)
            error("Conflicting legacy and preferred time controls for " * string(family) * ": " * join(sources, ", ") * ". Use only " * join(preferred, ", ") * ".")
        end
    end
    return nothing
end

function convert_override_value(key::Symbol, value, key_name::AbstractString)
    key in (:B_paths, :v_paths) && return config_tuple3_strings(value, key_name)
    key == :injection_position && return config_tuple3_floats(value, key_name)
    key == :field_subset && return config_field_subset(value, key_name)
    key in (:precision, :trajectory_output_precision, :compute_precision) && return config_precision(value, key_name)
    key in (:boundary, :compute_backend, :particle_selection, :injection_mode, :injection_position_mode, :injection_position_unit) && return config_symbol(value, key_name)
    key == :lag_mode && return CombinedFullMod.parse_lag_mode(String(value))
    key == :dmumu_start_mode && return CombinedFullMod.parse_dmumu_start_mode(value)
    key == :mu_bin_abs && return config_bool(value, key_name)
    key == :n_particles_to_use && return config_maybe_particle_count(value, key_name)
    key == :max_lag_steps && return config_maybe_int(value, key_name)
    key in (:trajectory_duration_gyroperiods, :trajectory_save_interval_gyroperiods, :integration_steps_per_gyroperiod, :lag_min_gyroperiods, :lag_max_gyroperiods, :lag_stride_gyroperiods) && return config_float(value, key_name)
    key == :use_usetex && return config_bool(value, key_name)
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
    generic_keys = Set(["h5_dir", "label", "medium", "file_stem", "total_file", "mode_file_pattern", "mode_decomposition_available", "available_modes"])
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
        haskey(input, "mode_decomposition_available") && (cfg[:mode_decomposition_available] = config_bool(input["mode_decomposition_available"], "[input].mode_decomposition_available"))
        haskey(input, "available_modes") && (cfg[:available_modes] = config_available_modes(input["available_modes"], "[input].available_modes"))
        return nothing
    end

    haskey(input, "layout") && (cfg[:input_layout] = parse_input_layout(input["layout"]))
    haskey(input, "mode_decomposition_root") && (cfg[:input_paths][:mode_decomposition_root] = resolve_repo_path(input["mode_decomposition_root"]))
    haskey(input, "mhd512_total_h5") && (cfg[:input_paths][:mhd512_total_h5] = resolve_repo_path(input["mhd512_total_h5"]))
    haskey(input, "mhd512_mode_dir") && (cfg[:input_paths][:mhd512_mode_dir] = resolve_repo_path(input["mhd512_mode_dir"]))
    return nothing
end

function apply_dmumu_section!(cfg, section)
    allowed_keys = union(Set(string(key) for key in keys(cfg[:dmumu_overrides])), Set(["particles"]))
    validate_toml_keys(section, allowed_keys, "[dmumu]")
    reject_duplicate_time_controls(section, ["min_lag_steps", "max_lag_steps", "lag_step_stride"], ["lag_min_gyroperiods", "lag_max_gyroperiods", "lag_stride_gyroperiods"], "[dmumu]", "lag_min_gyroperiods/lag_max_gyroperiods/lag_stride_gyroperiods")
    record_time_keys!(cfg, :dmumu_lag_min, :toml, [key for key in ("min_lag_steps", "lag_min_gyroperiods") if haskey(section, key)])
    record_time_keys!(cfg, :dmumu_lag_max, :toml, [key for key in ("max_lag_steps", "lag_max_gyroperiods") if haskey(section, key)])
    record_time_keys!(cfg, :dmumu_lag_stride, :toml, [key for key in ("lag_step_stride", "lag_stride_gyroperiods") if haskey(section, key)])

    override_section = Dict{String, Any}(String(key) => value for (key, value) in section if String(key) != "particles")
    apply_override_section!(cfg[:dmumu_overrides], override_section, "[dmumu]")

    haskey(section, "particles") && (cfg[:dmumu_particles] = config_dmumu_particles(section["particles"], "[dmumu].particles"))
    return nothing
end

function apply_dpp_section!(cfg, section)
    allowed_keys = Set(string(key) for key in keys(cfg[:dpp_overrides]))
    validate_toml_keys(section, allowed_keys, "[dpp]")
    reject_duplicate_time_controls(section, ["min_lag_steps", "max_lag_steps", "lag_step_stride"], ["lag_min_gyroperiods", "lag_max_gyroperiods", "lag_stride_gyroperiods"], "[dpp]", "lag_min_gyroperiods/lag_max_gyroperiods/lag_stride_gyroperiods")
    record_time_keys!(cfg, :dpp_lag_min, :toml, [key for key in ("min_lag_steps", "lag_min_gyroperiods") if haskey(section, key)])
    record_time_keys!(cfg, :dpp_lag_max, :toml, [key for key in ("max_lag_steps", "lag_max_gyroperiods") if haskey(section, key)])
    record_time_keys!(cfg, :dpp_lag_stride, :toml, [key for key in ("lag_step_stride", "lag_stride_gyroperiods") if haskey(section, key)])
    apply_override_section!(cfg[:dpp_overrides], section, "[dpp]")
    return nothing
end

function apply_toml_config!(cfg, config_path::AbstractString)
    absolute_config_path = resolve_cli_path(config_path)
    isfile(absolute_config_path) || error("Config file not found: " * absolute_config_path)
    config = TOML.parsefile(absolute_config_path)

    validate_toml_keys(config, Set(["input", "output", "run", "particles", "dmumu", "dpp", "trajectory", "time_reference"]), "config")

    input = toml_section(config, "input")
    apply_input_section!(cfg, input)

    output = toml_section(config, "output")
    validate_toml_keys(output, Set(["root"]), "[output]")
    haskey(output, "root") && (cfg[:output_root] = resolve_repo_path(output["root"]))

    time_reference = toml_section(config, "time_reference")
    validate_toml_keys(time_reference, Set(["mode"]), "[time_reference]")
    if haskey(time_reference, "mode")
        mode = replace(lowercase(strip(String(time_reference["mode"]))), "_" => "-")
        mode == "total-field-mean-magnitude" || error("[time_reference].mode currently supports only total-field-mean-magnitude.")
        cfg[:time_reference_mode] = mode
    end

    run = toml_section(config, "run")
    validate_toml_keys(
        run,
        Set([
            "cache_mode",
            "compute_dmumu",
            "compute_dpp",
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
    haskey(run, "compute_dpp") && (cfg[:compute_dpp] = config_bool(run["compute_dpp"], "[run].compute_dpp"))
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

    trajectory_section = toml_section(config, "trajectory")
    particles_section = toml_section(config, "particles")
    record_time_keys!(cfg, :trajectory_duration, :toml, [key for key in ("tOmega0_max", "trajectory_duration_gyroperiods") if haskey(trajectory_section, key) || haskey(particles_section, key)])
    record_time_keys!(cfg, :integration_resolution, :toml, [key for key in ("eta", "integration_steps_per_gyroperiod") if haskey(trajectory_section, key) || haskey(particles_section, key)])
    record_time_keys!(cfg, :trajectory_save_interval, :toml, [key for key in ("trajectory_time_stride", "trajectory_save_interval_gyroperiods") if haskey(trajectory_section, key) || haskey(particles_section, key)])
    reject_duplicate_time_controls(trajectory_section, ["tOmega0_max"], ["trajectory_duration_gyroperiods"], "[trajectory]", "trajectory_duration_gyroperiods")
    reject_duplicate_time_controls(trajectory_section, ["eta"], ["integration_steps_per_gyroperiod"], "[trajectory]", "integration_steps_per_gyroperiod")
    reject_duplicate_time_controls(trajectory_section, ["trajectory_time_stride"], ["trajectory_save_interval_gyroperiods"], "[trajectory]", "trajectory_save_interval_gyroperiods")
    reject_duplicate_time_controls(particles_section, ["tOmega0_max"], ["trajectory_duration_gyroperiods"], "[particles]", "trajectory_duration_gyroperiods")
    reject_duplicate_time_controls(particles_section, ["eta"], ["integration_steps_per_gyroperiod"], "[particles]", "integration_steps_per_gyroperiod")
    reject_duplicate_time_controls(particles_section, ["trajectory_time_stride"], ["trajectory_save_interval_gyroperiods"], "[particles]", "trajectory_save_interval_gyroperiods")
    apply_override_section!(cfg[:trajectory_overrides], trajectory_section, "[trajectory]")
    apply_override_section!(cfg[:trajectory_overrides], particles_section, "[particles]")
    apply_dmumu_section!(cfg, toml_section(config, "dmumu"))
    apply_dpp_section!(cfg, toml_section(config, "dpp"))
    return absolute_config_path
end

function apply_all_particles_dmumu!(cfg)
    cfg[:run_all_particles_dmumu] = true
    merge!(cfg[:dmumu_overrides], Dict{Symbol, Any}(
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
    cfg[:dmumu_overrides] = shallow_copy_dict(CACHE_PIPELINE_CFG[:dmumu_overrides])
    cfg[:dpp_overrides] = shallow_copy_dict(CACHE_PIPELINE_CFG[:dpp_overrides])
    cfg[:input_kind] = :legacy
    cfg[:input_spec] = default_input_spec()
    cfg[:input_paths] = default_input_paths()
    cfg[:output_root] = joinpath(PIPELINE_ROOT, "outputs", "campaigns_cache")
    cfg[:input_layout] = :mp_weakb
    cfg[:mode_decomposition_available] = false
    cfg[:available_modes] = String.(MODE_NAMES)
    cfg[:dmumu_particles] = :sample
    cfg[:smoke] = false
    cfg[:time_reference_mode] = "total-field-mean-magnitude"

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
        elseif argument == "--compute-dpp"
            cfg[:compute_dpp] = true
        elseif argument == "--no-compute-dpp"
            cfg[:compute_dpp] = false
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
    if cfg[:compute_dpp]
        cache_mode = :phase_space
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
            :trajectory_duration_gyroperiods => nothing,
            :tOmega0_max => 20.0,
            :trajectory_save_interval_gyroperiods => nothing,
            :trajectory_time_stride => 1,
            :progress_every => 20,
        ))
        merge!(cfg[:dmumu_overrides], Dict{Symbol, Any}(
            :field_subset => (16, 16, 16),
            :particle_chunk_size => 32,
            :first_particle => 1,
            :n_particles_to_use => 64,
            :particle_selection => :range,
            :dmumu_start_mode => :sliding,
            :lag_min_gyroperiods => nothing,
            :lag_max_gyroperiods => nothing,
            :lag_stride_gyroperiods => nothing,
            :min_lag_steps => 1,
            :n_lag_samples => 8,
            :max_lag_steps => 20,
            :n_mu_bins => 8,
            :min_count_per_cell => 3,
        ))
        merge!(cfg[:dpp_overrides], Dict{Symbol, Any}(
            :particle_chunk_size => 32,
            :first_particle => 1,
            :n_particles_to_use => 64,
            :particle_selection => :range,
            :lag_min_gyroperiods => nothing,
            :lag_max_gyroperiods => nothing,
            :lag_stride_gyroperiods => nothing,
            :min_lag_steps => 1,
            :n_lag_samples => 8,
            :max_lag_steps => 20,
            :n_energy_snapshots => 4,
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
        elseif startswith(argument, "--trajectory-duration-gyroperiods=")
            cfg[:trajectory_overrides][:trajectory_duration_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :trajectory_duration, :cli, ["trajectory_duration_gyroperiods"])
        elseif startswith(argument, "--integration-steps-per-gyroperiod=")
            cfg[:trajectory_overrides][:integration_steps_per_gyroperiod] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :integration_resolution, :cli, ["integration_steps_per_gyroperiod"])
        elseif startswith(argument, "--trajectory-save-interval-gyroperiods=")
            cfg[:trajectory_overrides][:trajectory_save_interval_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :trajectory_save_interval, :cli, ["trajectory_save_interval_gyroperiods"])
        elseif startswith(argument, "--dmumu-start-mode=")
            cfg[:dmumu_overrides][:dmumu_start_mode] = CombinedFullMod.parse_dmumu_start_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-lag-min-gyroperiods=")
            cfg[:dmumu_overrides][:lag_min_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dmumu_lag_min, :cli, ["lag_min_gyroperiods"])
        elseif startswith(argument, "--dmumu-lag-max-gyroperiods=")
            cfg[:dmumu_overrides][:lag_max_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dmumu_lag_max, :cli, ["lag_max_gyroperiods"])
        elseif startswith(argument, "--dmumu-lag-stride-gyroperiods=")
            cfg[:dmumu_overrides][:lag_stride_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dmumu_lag_stride, :cli, ["lag_stride_gyroperiods"])
        elseif startswith(argument, "--dmumu-min-lag-steps=")
            cfg[:dmumu_overrides][:min_lag_steps] = parse(Int, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dmumu_lag_min, :cli, ["min_lag_steps"])
        elseif startswith(argument, "--dmumu-lag-mode=")
            cfg[:dmumu_overrides][:lag_mode] = CombinedFullMod.parse_lag_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-n-lag-samples=")
            cfg[:dmumu_overrides][:n_lag_samples] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-lag-step-stride=")
            cfg[:dmumu_overrides][:lag_step_stride] = parse(Int, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dmumu_lag_stride, :cli, ["lag_step_stride"])
        elseif startswith(argument, "--dmumu-max-lag-steps=")
            cfg[:dmumu_overrides][:max_lag_steps] = config_maybe_int(split(argument, "=", limit=2)[2], "--dmumu-max-lag-steps")
            record_time_keys!(cfg, :dmumu_lag_max, :cli, ["max_lag_steps"])
        elseif startswith(argument, "--dmumu-particle-chunk-size=")
            cfg[:dmumu_overrides][:particle_chunk_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-first-particle=")
            cfg[:dmumu_overrides][:first_particle] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-n-particles=")
            cfg[:dmumu_overrides][:n_particles_to_use] = config_maybe_particle_count(split(argument, "=", limit=2)[2], "--dmumu-n-particles")
        elseif startswith(argument, "--dmumu-particle-selection=")
            cfg[:dmumu_overrides][:particle_selection] = CombinedFullMod.parse_particle_selection(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-particle-seed=")
            cfg[:dmumu_overrides][:particle_seed] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-particle-block-size=")
            cfg[:dmumu_overrides][:particle_block_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-n-mu-bins=")
            cfg[:dmumu_overrides][:n_mu_bins] = parse(Int, split(argument, "=", limit=2)[2])
        elseif argument == "--dmumu-mu-bin-abs"
            cfg[:dmumu_overrides][:mu_bin_abs] = true
        elseif argument == "--no-dmumu-mu-bin-abs" || argument == "--dmumu-signed-mu-bins"
            cfg[:dmumu_overrides][:mu_bin_abs] = false
        elseif startswith(argument, "--dmumu-mu-bin-abs=")
            cfg[:dmumu_overrides][:mu_bin_abs] = config_bool(split(argument, "=", limit=2)[2], "--dmumu-mu-bin-abs")
        elseif startswith(argument, "--dmumu-mu-min=")
            cfg[:dmumu_overrides][:mu_min] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dmumu-mu-max=")
            cfg[:dmumu_overrides][:mu_max] = parse(Float64, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-lag-min-gyroperiods=")
            cfg[:dpp_overrides][:lag_min_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dpp_lag_min, :cli, ["lag_min_gyroperiods"])
        elseif startswith(argument, "--dpp-lag-max-gyroperiods=")
            cfg[:dpp_overrides][:lag_max_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dpp_lag_max, :cli, ["lag_max_gyroperiods"])
        elseif startswith(argument, "--dpp-lag-stride-gyroperiods=")
            cfg[:dpp_overrides][:lag_stride_gyroperiods] = parse(Float64, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dpp_lag_stride, :cli, ["lag_stride_gyroperiods"])
        elseif startswith(argument, "--dpp-min-lag-steps=")
            cfg[:dpp_overrides][:min_lag_steps] = parse(Int, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dpp_lag_min, :cli, ["min_lag_steps"])
        elseif startswith(argument, "--dpp-lag-mode=")
            cfg[:dpp_overrides][:lag_mode] = DppFullMod.parse_lag_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-n-lag-samples=")
            cfg[:dpp_overrides][:n_lag_samples] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-lag-step-stride=")
            cfg[:dpp_overrides][:lag_step_stride] = parse(Int, split(argument, "=", limit=2)[2])
            record_time_keys!(cfg, :dpp_lag_stride, :cli, ["lag_step_stride"])
        elseif startswith(argument, "--dpp-max-lag-steps=")
            cfg[:dpp_overrides][:max_lag_steps] = config_maybe_int(split(argument, "=", limit=2)[2], "--dpp-max-lag-steps")
            record_time_keys!(cfg, :dpp_lag_max, :cli, ["max_lag_steps"])
        elseif startswith(argument, "--dpp-particle-chunk-size=")
            cfg[:dpp_overrides][:particle_chunk_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-first-particle=")
            cfg[:dpp_overrides][:first_particle] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-n-particles=")
            cfg[:dpp_overrides][:n_particles_to_use] = config_maybe_particle_count(split(argument, "=", limit=2)[2], "--dpp-n-particles")
        elseif startswith(argument, "--dpp-particle-selection=")
            cfg[:dpp_overrides][:particle_selection] = DppFullMod.parse_particle_selection(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-particle-seed=")
            cfg[:dpp_overrides][:particle_seed] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-particle-block-size=")
            cfg[:dpp_overrides][:particle_block_size] = parse(Int, split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--dpp-n-energy-snapshots=")
            cfg[:dpp_overrides][:n_energy_snapshots] = parse(Int, split(argument, "=", limit=2)[2])
        end
    end
    cli_requested_campaign_selection && (campaign_requests = nothing)
    reject_recorded_time_conflicts(cfg)

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
    println("  compute D_pp             = ", cfg[:compute_dpp])
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
