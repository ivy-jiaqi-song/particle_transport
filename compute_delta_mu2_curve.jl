using HDF5
using CSV
using CUDA
using DataFrames
using PyPlot
using Statistics

const UGAUSS_TO_T = 1e-10
const PC_TO_M = 3.085677581e16

const CFG = Dict{Symbol, Any}(
    :trajectory_h5 => joinpath(@__DIR__, "outputs", "campaigns", "0_5", "trajectory_cache","phase_space_10000000_GeV.h5"),
    :turbulence_h5 => raw"/data/multiphase/MP_WeakB_0_5tcs.h5",
    :B_paths => ("i_mag_field", "j_mag_field", "k_mag_field"),
    :box_length_pc => 200.0,
    :field_subset => nothing,
    :particle_chunk_size => 256,
    :max_lag_steps => nothing,      # nothing => use all available saved lags
    :lag_step_stride => 1,
    :lag_chunk_size => nothing,     # GPU only; nothing => process all selected lags at once
    :compute_backend => :gpu,      # :auto, :cpu, :gpu
    :compute_precision => Float32,  # field + mu reconstruction precision
    :gpu_threads => 256,
    :output_csv => nothing,         # optional duplicate table; set a path if you want CSV
    :output_h5 => joinpath(@__DIR__, "outputs", "campaigns", "0_5", "10000000_GeV", "delta_mu2_curve.h5"),
    :output_png => joinpath(@__DIR__, "outputs", "campaigns", "0_5", "10000000_GeV", "delta_mu2_curve.png"),
    :use_usetex => false,
)

function load_B_fields(cfg, ::Type{T}) where {T<:AbstractFloat}
    subset = cfg[:field_subset]
    h5open(cfg[:turbulence_h5], "r") do file
        function read_field(path)
            dataset = file[path]
            if subset === nothing
                return Array{T}(read(dataset))
            end
            nx, ny, nz = subset
            return Array{T}(dataset[1:nx, 1:ny, 1:nz])
        end

        Bx = read_field(cfg[:B_paths][1])
        By = read_field(cfg[:B_paths][2])
        Bz = read_field(cfg[:B_paths][3])
        Bx .*= T(UGAUSS_TO_T)
        By .*= T(UGAUSS_TO_T)
        Bz .*= T(UGAUSS_TO_T)
        return Bx, By, Bz
    end
end

function build_uniform_coords(cfg, nx, ny, nz, ::Type{T}) where {T<:AbstractFloat}
    box_length = T(cfg[:box_length_pc] * PC_TO_M)
    return collect(range(zero(T), box_length, length=nx)),
           collect(range(zero(T), box_length, length=ny)),
           collect(range(zero(T), box_length, length=nz))
end

@inline function trilinear_sample(arr, x, y, z, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
    fx = (x - xmin) / dx + 1
    fy = (y - ymin) / dy + 1
    fz = (z - zmin) / dz + 1

    i0 = clamp(floor(Int, fx), 1, nx - 1)
    j0 = clamp(floor(Int, fy), 1, ny - 1)
    k0 = clamp(floor(Int, fz), 1, nz - 1)

    tx = fx - i0
    ty = fy - j0
    tz = fz - k0

    c000 = arr[i0, j0, k0]
    c100 = arr[i0 + 1, j0, k0]
    c010 = arr[i0, j0 + 1, k0]
    c110 = arr[i0 + 1, j0 + 1, k0]
    c001 = arr[i0, j0, k0 + 1]
    c101 = arr[i0 + 1, j0, k0 + 1]
    c011 = arr[i0, j0 + 1, k0 + 1]
    c111 = arr[i0 + 1, j0 + 1, k0 + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz
end

@inline function dot3(ax, ay, az, bx, by, bz)
    return ax * bx + ay * by + az * bz
end

@inline function norm3(ax, ay, az)
    return sqrt(dot3(ax, ay, az, ax, ay, az))
end

@inline function pitch_mu_from_pB(px, py, pz, bx, by, bz)
    T = typeof(px + bx)
    pn = norm3(px, py, pz)
    bn = norm3(bx, by, bz)
    if !isfinite(pn) || !isfinite(bn) || pn == zero(T) || bn == zero(T)
        return T(NaN)
    end
    mu = dot3(px, py, pz, bx, by, bz) / (pn * bn)
    return clamp(mu, -one(T), one(T))
end

function reconstruct_mu_chunk_cpu(positions, momenta, Bx, By, Bz, xgrid, ygrid, zgrid, ::Type{T}) where {T<:AbstractFloat}
    n_particles = size(positions, 1)
    nsteps = size(positions, 3)
    mu = fill(T(NaN), nsteps, n_particles)

    xmin, xmax = xgrid[1], xgrid[end]
    ymin, ymax = ygrid[1], ygrid[end]
    zmin, zmax = zgrid[1], zgrid[end]
    dx = xgrid[2] - xgrid[1]
    dy = ygrid[2] - ygrid[1]
    dz = zgrid[2] - zgrid[1]
    nx, ny, nz = size(Bx)

    @inbounds for particle in 1:n_particles
        for step in 1:nsteps
            xx = T(positions[particle, 1, step])
            yy = T(positions[particle, 2, step])
            zz = T(positions[particle, 3, step])
            px = T(momenta[particle, 1, step])
            py = T(momenta[particle, 2, step])
            pz = T(momenta[particle, 3, step])

            if isnan(xx) || isnan(yy) || isnan(zz) || isnan(px) || isnan(py) || isnan(pz)
                continue
            end
            if xx < xmin || xx > xmax || yy < ymin || yy > ymax || zz < zmin || zz > zmax
                continue
            end

            bx = trilinear_sample(Bx, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            by = trilinear_sample(By, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            bz = trilinear_sample(Bz, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            mu[step, particle] = pitch_mu_from_pB(px, py, pz, bx, by, bz)
        end
    end

    return mu
end

function reconstruct_mu_kernel!(mu, positions, momenta, Bx, By, Bz, xmin, ymin, zmin, xmax, ymax, zmax, dx, dy, dz, nx, ny, nz)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nsteps = size(mu, 1)
    n_particles = size(mu, 2)
    ntotal = nsteps * n_particles

    if index > ntotal
        return
    end

    step = ((index - 1) % nsteps) + 1
    particle = ((index - 1) ÷ nsteps) + 1
    T = eltype(mu)

    xx = positions[particle, 1, step]
    yy = positions[particle, 2, step]
    zz = positions[particle, 3, step]
    px = momenta[particle, 1, step]
    py = momenta[particle, 2, step]
    pz = momenta[particle, 3, step]

    value = T(NaN)
    if !(isnan(xx) || isnan(yy) || isnan(zz) || isnan(px) || isnan(py) || isnan(pz))
        if xmin <= xx <= xmax && ymin <= yy <= ymax && zmin <= zz <= zmax
            bx = trilinear_sample(Bx, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            by = trilinear_sample(By, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            bz = trilinear_sample(Bz, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
            value = pitch_mu_from_pB(px, py, pz, bx, by, bz)
        end
    end

    mu[step, particle] = value
    return
end

function reconstruct_mu_chunk_gpu(positions, momenta, dBx, dBy, dBz, xgrid, ygrid, zgrid, cfg, ::Type{T}) where {T<:AbstractFloat}
    n_particles = size(positions, 1)
    nsteps = size(positions, 3)
    dpositions = CuArray(T.(positions))
    dmomenta = CuArray(T.(momenta))
    dmu = CUDA.fill(T(NaN), nsteps, n_particles)

    xmin, xmax = T(xgrid[1]), T(xgrid[end])
    ymin, ymax = T(ygrid[1]), T(ygrid[end])
    zmin, zmax = T(zgrid[1]), T(zgrid[end])
    dx = T(xgrid[2] - xgrid[1])
    dy = T(ygrid[2] - ygrid[1])
    dz = T(zgrid[2] - zgrid[1])
    nx, ny, nz = size(dBx)

    threads = Int(cfg[:gpu_threads])
    blocks = cld(length(dmu), threads)
    @cuda threads=threads blocks=blocks reconstruct_mu_kernel!(
        dmu,
        dpositions,
        dmomenta,
        dBx,
        dBy,
        dBz,
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        dx,
        dy,
        dz,
        nx,
        ny,
        nz,
    )
    CUDA.synchronize()
    return dmu
end

@inline function update_running_stats!(counts, means, m2s, lag_index, value)
    counts[lag_index] += 1
    delta = value - means[lag_index]
    means[lag_index] += delta / counts[lag_index]
    delta2 = value - means[lag_index]
    m2s[lag_index] += delta * delta2
end

function process_mu_chunk_cpu!(mu_chunk, lag_steps, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)
    nsteps, n_particles = size(mu_chunk)

    @inbounds for (lag_index, lag) in enumerate(lag_steps)
        lag >= nsteps && continue
        last_start = nsteps - lag
        for particle in 1:n_particles
            sumsq = 0.0
            count_pairs = 0
            for step in 1:last_start
                a = mu_chunk[step, particle]
                b = mu_chunk[step + lag, particle]
                if isnan(a) || isnan(b)
                    continue
                end
                delta_mu = b - a
                sumsq += delta_mu * delta_mu
                count_pairs += 1
            end
            if count_pairs > 0
                particle_value = sumsq / count_pairs
                update_running_stats!(particle_counts, particle_means, particle_m2s, lag_index, particle_value)
                pair_sum_squares[lag_index] += sumsq
                pair_counts[lag_index] += count_pairs
            end
        end
    end

    return nothing
end

function particle_lag_delta_mu2_kernel!(particle_values, particle_pair_counts, mu, lag_steps)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nlags = size(particle_values, 1)
    n_particles = size(particle_values, 2)
    ntotal = nlags * n_particles

    if index > ntotal
        return
    end

    lag_index = ((index - 1) % nlags) + 1
    particle = ((index - 1) ÷ nlags) + 1
    lag = lag_steps[lag_index]
    nsteps = size(mu, 1)
    T = eltype(particle_values)

    if lag >= nsteps
        particle_values[lag_index, particle] = T(NaN)
        particle_pair_counts[lag_index, particle] = Int32(0)
        return
    end

    sumsq = zero(T)
    count_pairs = Int32(0)
    last_start = nsteps - lag
    @inbounds for step in 1:last_start
        a = mu[step, particle]
        b = mu[step + lag, particle]
        if !(isnan(a) || isnan(b))
            delta_mu = b - a
            sumsq += delta_mu * delta_mu
            count_pairs += Int32(1)
        end
    end

    if count_pairs > 0
        particle_values[lag_index, particle] = sumsq / count_pairs
        particle_pair_counts[lag_index, particle] = count_pairs
    else
        particle_values[lag_index, particle] = T(NaN)
        particle_pair_counts[lag_index, particle] = Int32(0)
    end

    return
end

function lag_index_ranges(lag_steps, cfg)
    nlags = length(lag_steps)
    lag_chunk_size = cfg[:lag_chunk_size]
    if lag_chunk_size === nothing
        return [1:nlags]
    end

    size_value = Int(lag_chunk_size)
    size_value < 1 && error("lag_chunk_size must be >= 1")
    return [first_index:min(first_index + size_value - 1, nlags) for first_index in 1:size_value:nlags]
end

function aggregate_particle_lag_stats!(particle_values, particle_pair_counts, lag_offset, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)
    nlags_local, n_particles = size(particle_values)

    @inbounds for local_lag_index in 1:nlags_local
        global_lag_index = lag_offset + local_lag_index - 1
        for particle in 1:n_particles
            count_pairs = Int(particle_pair_counts[local_lag_index, particle])
            count_pairs <= 0 && continue
            particle_value = Float64(particle_values[local_lag_index, particle])
            isnan(particle_value) && continue
            update_running_stats!(particle_counts, particle_means, particle_m2s, global_lag_index, particle_value)
            pair_sum_squares[global_lag_index] += particle_value * count_pairs
            pair_counts[global_lag_index] += count_pairs
        end
    end

    return nothing
end

function process_mu_chunk_gpu!(dmu_chunk, lag_steps, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts, cfg)
    nsteps, n_particles = size(dmu_chunk)
    threads = Int(cfg[:gpu_threads])

    for lag_range in lag_index_ranges(lag_steps, cfg)
        lag_subset = Int32.(lag_steps[lag_range])
        dlag_subset = CuArray(lag_subset)
        nlags_local = length(lag_subset)
        dparticle_values = CUDA.fill(eltype(dmu_chunk)(NaN), nlags_local, n_particles)
        dparticle_pair_counts = CUDA.zeros(Int32, nlags_local, n_particles)

        blocks = cld(length(dparticle_values), threads)
        @cuda threads=threads blocks=blocks particle_lag_delta_mu2_kernel!(
            dparticle_values,
            dparticle_pair_counts,
            dmu_chunk,
            dlag_subset,
        )
        CUDA.synchronize()

        aggregate_particle_lag_stats!(
            Array(dparticle_values),
            Array(dparticle_pair_counts),
            first(lag_range),
            particle_counts,
            particle_means,
            particle_m2s,
            pair_sum_squares,
            pair_counts,
        )
    end

    return nothing
end

function build_lag_steps(nsteps::Integer, cfg)
    maxlag = cfg[:max_lag_steps] === nothing ? (nsteps - 1) : min(Int(cfg[:max_lag_steps]), nsteps - 1)
    stride = Int(cfg[:lag_step_stride])
    stride < 1 && error("lag_step_stride must be >= 1")
    lags = collect(1:stride:maxlag)
    isempty(lags) && error("No lag steps selected. Check max_lag_steps and lag_step_stride.")
    return lags
end

function sem_from_stats(count::Int, m2::Float64)
    if count <= 1
        return NaN
    end
    variance = m2 / (count - 1)
    return sqrt(variance / count)
end

function resolve_backend(cfg)
    requested = cfg[:compute_backend]
    if requested == :cpu
        return :cpu
    elseif requested == :gpu
        CUDA.functional() || error("CFG[:compute_backend] = :gpu but CUDA.functional() is false on this machine.")
        return :gpu
    elseif requested == :auto
        return CUDA.functional() ? :gpu : :cpu
    else
        error("Unknown compute backend: " * string(requested) * ". Use :auto, :cpu, or :gpu.")
    end
end

function build_output_dataframe(lag_steps, t_s, t_norm, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)
    tau_s = [t_s[lag + 1] - t_s[1] for lag in lag_steps]
    tau_norm = [t_norm[lag + 1] - t_norm[1] for lag in lag_steps]
    particle_std = [count <= 1 ? NaN : sqrt(particle_m2s[index] / (count - 1)) for (index, count) in enumerate(particle_counts)]
    particle_sem = [sem_from_stats(particle_counts[index], particle_m2s[index]) for index in eachindex(particle_counts)]
    pair_mean = [pair_counts[index] == 0 ? NaN : pair_sum_squares[index] / pair_counts[index] for index in eachindex(pair_counts)]

    return DataFrame(
        lag_step = lag_steps,
        tau_s = tau_s,
        tau_norm = tau_norm,
        delta_mu2_particle_mean = particle_means,
        delta_mu2_particle_std = particle_std,
        delta_mu2_particle_sem = particle_sem,
        delta_mu2_pair_mean = pair_mean,
        n_particles_used = particle_counts,
        n_pairs_used = pair_counts,
    )
end

function save_delta_mu2_h5(path_h5::AbstractString, df::DataFrame, cfg, backend::Symbol)
    mkpath(dirname(path_h5))
    h5open(path_h5, "w") do file
        for column_name in names(df)
            file[string(column_name)] = collect(df[!, column_name])
        end
        file["trajectory_h5"] = string(cfg[:trajectory_h5])
        file["turbulence_h5"] = string(cfg[:turbulence_h5])
        file["compute_backend"] = string(backend)
        file["compute_precision"] = string(cfg[:compute_precision])
        file["particle_chunk_size"] = Int(cfg[:particle_chunk_size])
        file["lag_step_stride"] = Int(cfg[:lag_step_stride])
        file["max_lag_steps"] = cfg[:max_lag_steps] === nothing ? -1 : Int(cfg[:max_lag_steps])
        file["lag_chunk_size"] = cfg[:lag_chunk_size] === nothing ? -1 : Int(cfg[:lag_chunk_size])
        file["box_length_pc"] = Float64(cfg[:box_length_pc])
        file["field_subset"] = cfg[:field_subset] === nothing ? "nothing" : string(cfg[:field_subset])
    end
end

function plot_delta_mu2(df::DataFrame, path_png::AbstractString; use_usetex::Bool=false)
    PyPlot.rc("text", usetex=use_usetex)
    figure(figsize=(7, 4))
    errorbar_stride = max(1, length(df.tau_norm) ÷ 25)
    errorbar_indices = collect(1:errorbar_stride:length(df.tau_norm))
    std_lower = df.delta_mu2_particle_mean .- df.delta_mu2_particle_std
    std_upper = df.delta_mu2_particle_mean .+ df.delta_mu2_particle_std
    fill_between(
        df.tau_norm,
        std_lower,
        std_upper,
        color="0.65",
        alpha=0.35,
        linewidth=0.0,
        zorder=1,
        label=raw"$\pm 1\sigma$ particle spread",
    )
    plot(
        df.tau_norm,
        df.delta_mu2_particle_mean,
        color="black",
        linewidth=1.6,
        zorder=2,
        label="particle mean",
    )
    errorbar(
        df.tau_norm[errorbar_indices],
        df.delta_mu2_particle_mean[errorbar_indices],
        yerr=df.delta_mu2_particle_sem[errorbar_indices],
        fmt="none",
        ecolor="tab:blue",
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.75,
        zorder=3,
        label="SEM of mean (sparse)",
    )
    xlabel(raw"$\tau\Omega_0$")
    ylabel(raw"$\langle (\Delta\mu)^2 \rangle$")
    grid(true, alpha=0.3)
    legend(frameon=false, loc="best")
    tight_layout()
    savefig(path_png, dpi=200)
    close("all")
end

function validate_trajectory_layout(positions_ds, momenta_ds, t_s, t_norm)
    size(positions_ds) == size(momenta_ds) || error("positions and momenta have different shapes in trajectory HDF5.")
    nsteps = size(positions_ds, 3)
    length(t_s) == nsteps || error("Trajectory HDF5 mismatch: positions store $nsteps saved steps but t_s has $(length(t_s)) values.")
    length(t_norm) == nsteps || error("Trajectory HDF5 mismatch: positions store $nsteps saved steps but t_norm has $(length(t_norm)) values.")
    return nsteps
end

function run_delta_mu2_curve(cfg)
    mkpath(dirname(cfg[:output_h5]))
    mkpath(dirname(cfg[:output_png]))
    if cfg[:output_csv] !== nothing
        mkpath(dirname(cfg[:output_csv]))
    end

    backend = resolve_backend(cfg)
    T = cfg[:compute_precision]

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

    h5open(cfg[:trajectory_h5], "r") do file
        positions_ds = file["positions"]
        momenta_ds = file["momenta"]
        t_norm = Float64.(read(file["t_norm"]))
        t_s = Float64.(read(file["t_s"]))
        nsteps = validate_trajectory_layout(positions_ds, momenta_ds, t_s, t_norm)
        n_particles, _, _ = size(positions_ds)

        lag_steps = build_lag_steps(nsteps, cfg)
        nlags = length(lag_steps)

        particle_counts = zeros(Int, nlags)
        particle_means = zeros(Float64, nlags)
        particle_m2s = zeros(Float64, nlags)
        pair_sum_squares = zeros(Float64, nlags)
        pair_counts = zeros(Int, nlags)

        chunk_size = min(Int(cfg[:particle_chunk_size]), n_particles)
        nchunks = cld(n_particles, chunk_size)

        println("Processing trajectory file: ", cfg[:trajectory_h5])
        println("  backend         = ", backend)
        println("  precision       = ", T)
        println("  particles       = ", n_particles)
        println("  saved steps     = ", nsteps)
        println("  lag count       = ", nlags)
        println("  particle chunk  = ", chunk_size)
        println("  chunks          = ", nchunks)
        if backend == :gpu
            println("  lag chunk       = ", cfg[:lag_chunk_size] === nothing ? "all selected lags" : cfg[:lag_chunk_size])
        end

        for chunk_id in 1:nchunks
            first_particle = (chunk_id - 1) * chunk_size + 1
            last_particle = min(chunk_id * chunk_size, n_particles)
            println("Chunk ", chunk_id, "/", nchunks, ": particles ", first_particle, "-", last_particle)

            positions = positions_ds[first_particle:last_particle, :, :]
            momenta = momenta_ds[first_particle:last_particle, :, :]

            if backend == :gpu
                dmu_chunk = reconstruct_mu_chunk_gpu(positions, momenta, dBx, dBy, dBz, xgrid, ygrid, zgrid, cfg, T)
                process_mu_chunk_gpu!(dmu_chunk, lag_steps, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts, cfg)
                GC.gc(false)
                CUDA.reclaim()
            else
                mu_chunk = reconstruct_mu_chunk_cpu(positions, momenta, Bx, By, Bz, xgrid, ygrid, zgrid, T)
                process_mu_chunk_cpu!(mu_chunk, lag_steps, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)
            end
        end

        out_df = build_output_dataframe(lag_steps, t_s, t_norm, particle_counts, particle_means, particle_m2s, pair_sum_squares, pair_counts)

        if cfg[:output_csv] !== nothing
            CSV.write(cfg[:output_csv], out_df)
            println("Saved curve CSV to ", cfg[:output_csv])
        end

        save_delta_mu2_h5(cfg[:output_h5], out_df, cfg, backend)
        println("Saved curve HDF5 to ", cfg[:output_h5])

        plot_delta_mu2(out_df, cfg[:output_png]; use_usetex=cfg[:use_usetex])
        println("Saved curve plot to ", cfg[:output_png])
    end

    return nothing
end

function main()
    run_delta_mu2_curve(CFG)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
