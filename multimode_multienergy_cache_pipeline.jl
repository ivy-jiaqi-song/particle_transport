haskey(ENV, "MPLCONFIGDIR") || (ENV["MPLCONFIGDIR"] = "/tmp/mpl")

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

const MODE_DECOMPOSITION_ROOT = raw"/home/user0001/MHDFlows_replicate/multiphase_mode_decomposition/outputs"
const TURBULENCE_TAGS = ("0_5", "0_9", "1_0")
const MODE_NAMES = ("alfven", "fast", "slow")

const MHD512_DATASET_TAG = "512_a_00100"
const MHD512_TOTAL_H5 = raw"/home/user0001/MHDFlows_replicate/h5_outputs/512_a.00100.h5"
const MHD512_MODE_DIR = raw"/home/user0001/MHDFlows_replicate/mhdflows512_mode_decomposition/outputs/512_a_00100_cs10_L200/mode_h5"

const CACHE_PIPELINE_CFG = Dict{Symbol, Any}(
    :cache_mode => :mu,
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
        :use_usetex => false,
    ),
)

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

function mode_h5_path(turbulence_tag::AbstractString, mode_name::AbstractString)
    full_tag = "MP_WeakB_" * turbulence_tag * "tcs"
    return joinpath(
        MODE_DECOMPOSITION_ROOT,
        full_tag * "_cs1_L200",
        "mode_h5",
        full_tag * "_" * mode_name * ".h5",
    )
end

function build_mode_campaigns(root_prefix::AbstractString)
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
                    :turbulence_h5 => mode_h5_path(turbulence_tag, mode_name),
                ),
            )
        end
    end
    return campaigns
end

function mhd512_mode_h5_path(mode_name::AbstractString)
    return joinpath(MHD512_MODE_DIR, MHD512_DATASET_TAG * "_" * mode_name * ".h5")
end

function build_mhd512_campaigns(root_prefix::AbstractString)
    campaigns = Dict{Symbol, Any}[]
    push!(
        campaigns,
        Dict{Symbol, Any}(
            :campaign_tag => MHD512_DATASET_TAG * "/total",
            :turbulence_tag => MHD512_DATASET_TAG,
            :mode_name => "total",
            :campaign_root => joinpath(root_prefix, MHD512_DATASET_TAG, "total"),
            :turbulence_h5 => MHD512_TOTAL_H5,
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
                :turbulence_h5 => mhd512_mode_h5_path(mode_name),
            ),
        )
    end
    return campaigns
end

function build_campaigns(input_layout::Symbol, root_prefix::AbstractString)
    input_layout == :mp_weakb && return build_mode_campaigns(root_prefix)
    input_layout == :mhd512 && return build_mhd512_campaigns(root_prefix)
    error("Unknown input layout: " * string(input_layout))
end

function validate_selected_values(campaigns, selected_values, key::Symbol, flag_name::AbstractString)
    selected_values === nothing && return nothing
    available = unique([campaign[key] for campaign in campaigns])
    missing = [value for value in selected_values if !(value in available)]
    isempty(missing) && return nothing
    error("Unknown " * flag_name * " value(s): " * join(missing, ", ") * ". Available: " * join(available, ", "))
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
        filtered = [campaign for campaign in filtered if campaign[:campaign_tag] in wanted]
    end

    isempty(filtered) && error("No campaigns selected. Check --turbulence, --mode, and --campaign.")
    return filtered
end

function validate_mode_files(campaigns)
    missing = [campaign[:turbulence_h5] for campaign in campaigns if !isfile(campaign[:turbulence_h5])]
    isempty(missing) && return nothing
    error("Missing mode HDF5 file(s):\n" * join(missing, "\n"))
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

function compact_outputs_complete(paths)
    try
        verify_combined_outputs(paths.combined_h5, paths.delta_png, paths.dmumu_heatmap_png, paths.dmumu_collapsed_png)
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
        println("Mu bins: ", n_bins)
        println("Particle chunk: ", chunk_size)
        println("Chunks: ", nchunks)
        println("Full-pair accumulation backend: cpu")
        println("Julia threads: ", Threads.nthreads(), " active pool, max thread id ", Threads.maxthreadid())

        for chunk_id in 1:nchunks
            selection_first = (chunk_id - 1) * chunk_size + 1
            selection_last = min(selected_particle_count, selection_first + chunk_size - 1)
            chunk_indices = particle_indices[selection_first:selection_last]
            println("Chunk ", chunk_id, "/", nchunks, ": ", length(chunk_indices), " selected particles, index span ", first(chunk_indices), "-", last(chunk_indices))

            mu_batch = read_mu_batch(mu_dataset, chunk_indices)
            mu_chunk = permutedims(mu_batch, (2, 1))

            CombinedFullMod.process_mu_chunk_full_pairs_cpu!(
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

        CombinedFullMod.plot_dmumu_heatmap_full(cfg[:output_heatmap_png], mu_edges, tau_norm, dmumu_centered_norm; use_usetex=cfg[:use_usetex])
        println("Saved D_mumu heatmap to ", cfg[:output_heatmap_png])

        CombinedFullMod.plot_collapsed_dmumu_full(
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

function run_energy_pipeline!(cfg, fields, base_runner_cfg, energy_GeV)
    paths = build_energy_paths(cfg, energy_GeV)
    mkpath(paths.cache_dir)
    mkpath(paths.science_dir)

    println()
    println("=== Energy ", energy_GeV, " GeV ===")
    if cfg[:skip_completed_outputs] && compact_outputs_complete(paths)
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

    analysis_cfg = merge_cfg(
        CombinedFullMod.COMBINED_FULL_CFG,
        merge_cfg(
            cfg[:combined_overrides],
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
        :energies => cfg[:energies],
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

function runtime_config()
    cfg = shallow_copy_dict(CACHE_PIPELINE_CFG)
    cfg[:trajectory_overrides] = shallow_copy_dict(CACHE_PIPELINE_CFG[:trajectory_overrides])
    cfg[:combined_overrides] = shallow_copy_dict(CACHE_PIPELINE_CFG[:combined_overrides])

    cache_mode = CACHE_PIPELINE_CFG[:cache_mode]
    input_layout = :mp_weakb
    for argument in ARGS
        if startswith(argument, "--cache-mode=")
            cache_mode = parse_cache_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--cache=")
            cache_mode = parse_cache_mode(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--layout=")
            input_layout = parse_input_layout(split(argument, "=", limit=2)[2])
        elseif startswith(argument, "--input-layout=")
            input_layout = parse_input_layout(split(argument, "=", limit=2)[2])
        elseif argument == "--mhd512"
            input_layout = :mhd512
        elseif argument == "--mp-weakb"
            input_layout = :mp_weakb
        end
    end
    cfg[:cache_mode] = cache_mode
    cfg[:input_layout] = input_layout

    root_prefix = joinpath(@__DIR__, "outputs", "campaigns_cache", cache_mode_label(cache_mode))
    if any(argument -> argument == "--smoke", ARGS)
        root_prefix = joinpath(@__DIR__, "outputs", "campaigns_cache", "smoke", cache_mode_label(cache_mode))
    end
    all_campaigns = build_campaigns(input_layout, root_prefix)

    selected_turbulences = nothing
    selected_modes = nothing
    selected_campaigns = nothing

    if any(argument -> argument == "--smoke", ARGS)
        cfg[:energies] = [1e5, 3e5]
        selected_campaigns = input_layout == :mhd512 ? [MHD512_DATASET_TAG * "/total"] : ["0_5/alfven"]
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
            :n_lag_samples => 8,
            :max_lag_steps => 20,
            :n_mu_bins => 8,
            :min_count_per_cell => 3,
        ))
    end

    for argument in ARGS
        if startswith(argument, "--turbulence=")
            selected_turbulences = split_csv_selector(split(argument, "=", limit=2)[2]; flag_name="--turbulence")
        elseif startswith(argument, "--mode=")
            selected_modes = split_csv_selector(split(argument, "=", limit=2)[2]; flag_name="--mode")
        elseif startswith(argument, "--campaign=")
            selected_campaigns = parse_campaign_selector(split(argument, "=", limit=2)[2])
        end
    end

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
        cfg[:run_all_particles_dmumu] = true
        merge!(cfg[:combined_overrides], Dict{Symbol, Any}(
            :first_particle => 1,
            :n_particles_to_use => nothing,
            :particle_selection => :range,
            :allow_huge => true,
        ))
    end

    validate_selected_values(all_campaigns, selected_turbulences, :turbulence_tag, "--turbulence")
    validate_selected_values(all_campaigns, selected_modes, :mode_name, "--mode")
    validate_selected_values(all_campaigns, selected_campaigns, :campaign_tag, "--campaign")

    cfg[:mode_campaigns] = filter_campaigns(all_campaigns, selected_turbulences, selected_modes, selected_campaigns)
    validate_mode_files(cfg[:mode_campaigns])
    return cfg
end

function run_mode_campaigns(cfg)
    results = Dict{String, Any}()
    for campaign_spec in cfg[:mode_campaigns]
        println()
        println("=== Mode campaign ", campaign_spec[:campaign_tag], " ===")
        println("  mode H5                  = ", campaign_spec[:turbulence_h5])
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
    println("  input layout             = ", cfg[:input_layout])
    println("  mode campaigns           = ", campaign_tags(cfg))
    println("  cache mode               = ", cfg[:cache_mode])
    println("  energies [GeV]           = ", cfg[:energies])
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
