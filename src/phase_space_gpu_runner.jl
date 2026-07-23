using HDF5
using CUDA
using Random
using Statistics

include(joinpath(@__DIR__, "time_units.jl"))

const C_LIGHT = 2.99792458e8
const Q_E = 1.602176634e-19
const M_P = 1.67262192369e-27
const EV_TO_J = 1.602176634e-19
const GEV_TO_J = 1e9 * EV_TO_J
const UGAUSS_TO_T = 1e-10
const KMPS_TO_MPS = 1e3
const PC_TO_M = 3.085677581e16
const PIPELINE_ROOT = dirname(@__DIR__)
const DEFAULT_OUTPUT_DIR = joinpath(PIPELINE_ROOT, "outputs", "phase_space_gpu_runner")

const CFG = Dict{Symbol, Any}(
    :file => raw"/data/multiphase/MP_WeakB_0_5tcs.h5",
    :B_paths => ("i_mag_field", "j_mag_field", "k_mag_field"),
    :v_paths => ("i_velocity", "j_velocity", "k_velocity"),
    :velocity_unit_in_m_per_s => KMPS_TO_MPS,
    :box_length_pc => 200.0,
    :eta => 0.1,
    :integration_steps_per_gyroperiod => nothing,
    :tOmega0_max => 5000.0,
    :trajectory_duration_gyroperiods => nothing,
    :use_cfl => false,
    :cfl => 0.2,
    :seed => 42,
    :energies => [1e5],
    :n_particles => 100000,
    :precision => Float64,
    :field_subset => nothing,
    :boundary => :periodic,   # :open or :periodic
    :injection_mode => :isotropic,
    :injection_mu0 => 0.0,
    :injection_position_mode => :random,
    :injection_position => (0.5, 0.5, 0.5),
    :injection_position_unit => :box_fraction,
    :trajectory_time_stride => 1,
    :trajectory_save_interval_gyroperiods => nothing,
    :trajectory_output_precision => Float32,
    :progress_every => 5000,
    :output_dir => DEFAULT_OUTPUT_DIR,
)

function energy_to_gamma(E_GeV, ::Type{T}) where {T<:AbstractFloat}
    kinetic = T(E_GeV * GEV_TO_J)
    return kinetic / (T(M_P) * T(C_LIGHT)^2) + one(T)
end

function energy_to_speed(E_GeV, ::Type{T}) where {T<:AbstractFloat}
    gamma = energy_to_gamma(E_GeV, T)
    return T(C_LIGHT) * sqrt(one(T) - inv(gamma^2))
end

function rand_unit_vector(rng, ::Type{T}) where {T<:AbstractFloat}
    z = T(2 * rand(rng) - 1)
    phi = T(2 * pi * rand(rng))
    r = sqrt(max(zero(T), one(T) - z^2))
    return (r * cos(phi), r * sin(phi), z)
end

function injection_mode(cfg)
    mode = get(cfg, :injection_mode, :isotropic)
    mode = mode isa Symbol ? mode : Symbol(replace(lowercase(strip(String(mode))), "-" => "_"))
    mode in (:isotropic, :fixed_mu) || error("injection_mode must be isotropic or fixed-mu")
    return mode
end

function injection_position_mode(cfg)
    mode = get(cfg, :injection_position_mode, :random)
    mode = mode isa Symbol ? mode : Symbol(replace(lowercase(strip(String(mode))), "-" => "_"))
    mode in (:random, :fixed) || error("injection_position_mode must be random or fixed")
    return mode
end

function injection_position_unit(cfg)
    unit = get(cfg, :injection_position_unit, :box_fraction)
    unit = unit isa Symbol ? unit : Symbol(replace(lowercase(strip(String(unit))), "-" => "_"))
    unit in (:box_fraction, :m, :pc) || error("injection_position_unit must be box-fraction, m, or pc")
    return unit
end

function injection_position_tuple(cfg, ::Type{T}) where {T<:AbstractFloat}
    raw = get(cfg, :injection_position, (0.5, 0.5, 0.5))
    length(raw) == 3 || error("injection_position must contain exactly three values")
    return (T(raw[1]), T(raw[2]), T(raw[3]))
end

function load_static_fields(cfg, ::Type{T}) where {T<:AbstractFloat}
    subset = cfg[:field_subset]
    h5open(cfg[:file], "r") do f
        function read_field(path)
            data = f[path]
            if subset === nothing
                return Array{T}(read(data))
            end
            nx, ny, nz = subset
            return Array{T}(data[1:nx, 1:ny, 1:nz])
        end

        Bx = read_field(cfg[:B_paths][1])
        By = read_field(cfg[:B_paths][2])
        Bz = read_field(cfg[:B_paths][3])
        vx = read_field(cfg[:v_paths][1])
        vy = read_field(cfg[:v_paths][2])
        vz = read_field(cfg[:v_paths][3])

        Bx .*= T(UGAUSS_TO_T)
        By .*= T(UGAUSS_TO_T)
        Bz .*= T(UGAUSS_TO_T)
        vx .*= T(cfg[:velocity_unit_in_m_per_s])
        vy .*= T(cfg[:velocity_unit_in_m_per_s])
        vz .*= T(cfg[:velocity_unit_in_m_per_s])

        return Bx, By, Bz, vx, vy, vz
    end
end

function build_uniform_coords(cfg, nx, ny, nz, ::Type{T}) where {T<:AbstractFloat}
    L = T(cfg[:box_length_pc] * PC_TO_M)
    return collect(range(zero(T), L, length=nx)),
           collect(range(zero(T), L, length=ny)),
           collect(range(zero(T), L, length=nz))
end

function estimate_min_dx(x, y, z)
    return min(minimum(diff(x)), minimum(diff(y)), minimum(diff(z)))
end

function write_injection_metadata!(file, cfg)
    file["injection_mode"] = string(injection_mode(cfg))
    file["injection_mu0"] = Float64[get(cfg, :injection_mu0, 0.0)]
    file["injection_position_mode"] = string(injection_position_mode(cfg))
    file["injection_position"] = Float64[Float64(v) for v in injection_position_tuple(cfg, Float64)]
    file["injection_position_unit"] = string(injection_position_unit(cfg))
    return nothing
end

function fixed_injection_position(cfg, x, y, z, ::Type{T}) where {T<:AbstractFloat}
    xmin, xmax = x[1], x[end]
    ymin, ymax = y[1], y[end]
    zmin, zmax = z[1], z[end]
    px, py, pz = injection_position_tuple(cfg, T)
    unit = injection_position_unit(cfg)

    if unit == :box_fraction
        xi = xmin + px * (xmax - xmin)
        yi = ymin + py * (ymax - ymin)
        zi = zmin + pz * (zmax - zmin)
    elseif unit == :pc
        xi = px * T(PC_TO_M)
        yi = py * T(PC_TO_M)
        zi = pz * T(PC_TO_M)
    else
        xi, yi, zi = px, py, pz
    end

    if !(xmin <= xi <= xmax && ymin <= yi <= ymax && zmin <= zi <= zmax)
        error("fixed injection position is outside the simulation box")
    end
    return xi, yi, zi
end

function fixed_mu_unit_vector(rng, bx, by, bz, mu0, ::Type{T}) where {T<:AbstractFloat}
    bn = sqrt(bx * bx + by * by + bz * bz)
    (!isfinite(bn) || bn == zero(T)) && error("cannot use fixed-mu injection where local B is zero or non-finite")
    bhx, bhy, bhz = bx / bn, by / bn, bz / bn

    if abs(bhx) < T(0.9)
        e1x, e1y, e1z = zero(T), bhz, -bhy
    else
        e1x, e1y, e1z = -bhz, zero(T), bhx
    end
    e1n = sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x, e1y, e1z = e1x / e1n, e1y / e1n, e1z / e1n
    e2x = bhy * e1z - bhz * e1y
    e2y = bhz * e1x - bhx * e1z
    e2z = bhx * e1y - bhy * e1x

    phi = T(2 * pi * rand(rng))
    perp = sqrt(max(zero(T), one(T) - mu0 * mu0))
    cphi, sphi = cos(phi), sin(phi)
    return (
        mu0 * bhx + perp * (cphi * e1x + sphi * e2x),
        mu0 * bhy + perp * (cphi * e1y + sphi * e2y),
        mu0 * bhz + perp * (cphi * e1z + sphi * e2z),
    )
end

function init_particles(cfg, x, y, z, gamma0, v0, ::Type{T}, Bx=nothing, By=nothing, Bz=nothing) where {T<:AbstractFloat}
    rng = MersenneTwister(cfg[:seed])
    n = cfg[:n_particles]
    x1 = Vector{T}(undef, n)
    x2 = Vector{T}(undef, n)
    x3 = Vector{T}(undef, n)
    p1 = Vector{T}(undef, n)
    p2 = Vector{T}(undef, n)
    p3 = Vector{T}(undef, n)
    xmin, xmax = x[1], x[end]
    ymin, ymax = y[1], y[end]
    zmin, zmax = z[1], z[end]
    dx_grid = x[2] - x[1]
    dy_grid = y[2] - y[1]
    dz_grid = z[2] - z[1]
    nx, ny, nz = length(x), length(y), length(z)
    pscale = gamma0 * T(M_P) * v0
    pos_mode = injection_position_mode(cfg)
    mom_mode = injection_mode(cfg)
    fixed_pos = pos_mode == :fixed ? fixed_injection_position(cfg, x, y, z, T) : nothing
    mu0 = T(get(cfg, :injection_mu0, 0.0))
    if mom_mode == :fixed_mu
        -one(T) <= mu0 <= one(T) || error("injection_mu0 must be in [-1, 1]")
        (Bx === nothing || By === nothing || Bz === nothing) && error("fixed-mu injection requires magnetic field arrays")
    end

    for i in 1:n
        if fixed_pos === nothing
            x1[i] = rand(rng, T) * (xmax - xmin) + xmin
            x2[i] = rand(rng, T) * (ymax - ymin) + ymin
            x3[i] = rand(rng, T) * (zmax - zmin) + zmin
        else
            x1[i], x2[i], x3[i] = fixed_pos
        end

        if mom_mode == :fixed_mu
            bx = trilinear_sample(Bx, x1[i], x2[i], x3[i], xmin, ymin, zmin, dx_grid, dy_grid, dz_grid, nx, ny, nz)
            by = trilinear_sample(By, x1[i], x2[i], x3[i], xmin, ymin, zmin, dx_grid, dy_grid, dz_grid, nx, ny, nz)
            bz = trilinear_sample(Bz, x1[i], x2[i], x3[i], xmin, ymin, zmin, dx_grid, dy_grid, dz_grid, nx, ny, nz)
            ux, uy, uz = fixed_mu_unit_vector(rng, bx, by, bz, mu0, T)
        else
            ux, uy, uz = rand_unit_vector(rng, T)
        end
        p1[i] = pscale * ux
        p2[i] = pscale * uy
        p3[i] = pscale * uz
    end

    return x1, x2, x3, p1, p2, p3
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

@inline function cross3(ax, ay, az, bx, by, bz)
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
end

@inline function boris_push(px, py, pz, x, y, z, ex, ey, ez, bx, by, bz, dt)
    T = typeof(px)
    half = dt / T(2)
    qm = T(Q_E)
    mp = T(M_P)
    cl = T(C_LIGHT)

    pmx = px + qm * ex * half
    pmy = py + qm * ey * half
    pmz = pz + qm * ez * half

    gamma_minus = sqrt(one(T) + (pmx * pmx + pmy * pmy + pmz * pmz) / (mp^2 * cl^2))
    tx = qm * bx * dt / (T(2) * mp * gamma_minus)
    ty = qm * by * dt / (T(2) * mp * gamma_minus)
    tz = qm * bz * dt / (T(2) * mp * gamma_minus)

    t2 = tx * tx + ty * ty + tz * tz
    sx = T(2) * tx / (one(T) + t2)
    sy = T(2) * ty / (one(T) + t2)
    sz = T(2) * tz / (one(T) + t2)

    c1x, c1y, c1z = cross3(pmx, pmy, pmz, tx, ty, tz)
    ppx = pmx + c1x
    ppy = pmy + c1y
    ppz = pmz + c1z

    c2x, c2y, c2z = cross3(ppx, ppy, ppz, sx, sy, sz)
    ppx2 = pmx + c2x
    ppy2 = pmy + c2y
    ppz2 = pmz + c2z

    pxn = ppx2 + qm * ex * half
    pyn = ppy2 + qm * ey * half
    pzn = ppz2 + qm * ez * half

    gamma_new = sqrt(one(T) + (pxn * pxn + pyn * pyn + pzn * pzn) / (mp^2 * cl^2))
    vxn = pxn / (gamma_new * mp)
    vyn = pyn / (gamma_new * mp)
    vzn = pzn / (gamma_new * mp)

    return pxn, pyn, pzn, x + vxn * dt, y + vyn * dt, z + vzn * dt
end

@inline function wrap_periodic_coordinate(x, xmin, xmax)
    width = xmax - xmin
    return x - floor((x - xmin) / width) * width
end

function advance_particles_kernel!(x1_step, x2_step, x3_step, p1_step, p2_step, p3_step, x1, x2, x3, p1, p2, p3, alive, Bx, By, Bz, vx, vy, vz, dt, xmin, ymin, zmin, xmax, ymax, zmax, dx, dy, dz, nx, ny, nz, do_push, periodic_boundary)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    npart = length(alive)
    if p > npart
        return
    end

    if !alive[p]
        x1_step[p] = eltype(x1_step)(NaN)
        x2_step[p] = eltype(x2_step)(NaN)
        x3_step[p] = eltype(x3_step)(NaN)
        p1_step[p] = eltype(p1_step)(NaN)
        p2_step[p] = eltype(p2_step)(NaN)
        p3_step[p] = eltype(p3_step)(NaN)
        return
    end

    xx = x1[p]
    yy = x2[p]
    zz = x3[p]

    x1_step[p] = xx
    x2_step[p] = yy
    x3_step[p] = zz
    p1_step[p] = p1[p]
    p2_step[p] = p2[p]
    p3_step[p] = p3[p]

    if do_push
        bx = trilinear_sample(Bx, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        by = trilinear_sample(By, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        bz = trilinear_sample(Bz, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        vfx = trilinear_sample(vx, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        vfy = trilinear_sample(vy, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        vfz = trilinear_sample(vz, xx, yy, zz, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        ex, ey, ez = cross3(vfx, vfy, vfz, bx, by, bz)
        ex = -ex
        ey = -ey
        ez = -ez

        p1n, p2n, p3n, x1n, x2n, x3n = boris_push(p1[p], p2[p], p3[p], xx, yy, zz, ex, ey, ez, bx, by, bz, dt)

        if periodic_boundary
            x1n = wrap_periodic_coordinate(x1n, xmin, xmax)
            x2n = wrap_periodic_coordinate(x2n, ymin, ymax)
            x3n = wrap_periodic_coordinate(x3n, zmin, zmax)
        end

        p1[p] = p1n
        p2[p] = p2n
        p3[p] = p3n
        x1[p] = x1n
        x2[p] = x2n
        x3[p] = x3n

        if !periodic_boundary && (x1n < xmin || x1n > xmax || x2n < ymin || x2n > ymax || x3n < zmin || x3n > zmax)
            alive[p] = false
        end
    end

    return
end

function sampled_step_indices(nsteps::Integer, stride::Integer)
    stride < 1 && error("trajectory_time_stride must be >= 1")
    idx = collect(1:stride:nsteps)
    if isempty(idx) || idx[end] != nsteps
        push!(idx, nsteps)
    end
    return idx
end

function estimate_phase_space_bytes(n_particles::Integer, nsave::Integer, ::Type{T}) where {T}
    return Int128(6) * Int128(n_particles) * Int128(nsave) * Int128(sizeof(T))
end

function bytes_to_gib(bytes::Integer)
    return Float64(bytes) / 1024.0^3
end

function create_phase_space_writer(path::AbstractString, n_particles::Integer, nsave::Integer, ::Type{Tout}) where {Tout<:AbstractFloat}
    file = h5open(path, "w")
    positions = create_dataset(file, "positions", Tout, (n_particles, 3, nsave))
    momenta = create_dataset(file, "momenta", Tout, (n_particles, 3, nsave))
    return (file=file, positions=positions, momenta=momenta, outtype=Tout)
end

function write_phase_space_step!(writer, save_idx::Integer, x1_step, x2_step, x3_step, p1_step, p2_step, p3_step)
    Tout = writer.outtype
    writer.positions[:, 1, save_idx] = Tout.(x1_step)
    writer.positions[:, 2, save_idx] = Tout.(x2_step)
    writer.positions[:, 3, save_idx] = Tout.(x3_step)
    writer.momenta[:, 1, save_idx] = Tout.(p1_step)
    writer.momenta[:, 2, save_idx] = Tout.(p2_step)
    writer.momenta[:, 3, save_idx] = Tout.(p3_step)
end

function finalize_phase_space_writer!(writer, cfg, energy_GeV, t_norm_save, t_s_save, t_gyroperiods_save, alive_fraction_save, trajectory_time, save_time, Omega0, B0)
    file = writer.file
    file["t_norm"] = Float64.(t_norm_save)
    file["t_s"] = Float64.(t_s_save)
    file["t_gyroperiods"] = Float64.(t_gyroperiods_save)
    file["alive_fraction"] = Float64.(alive_fraction_save)
    file["energy_GeV"] = Float64[energy_GeV]
    file["n_particles"] = Int[cfg[:n_particles]]
    write_time_reference_metadata!(file, Omega0, B0)
    write_trajectory_time_metadata!(file, trajectory_time, save_time)
    file["boundary_mode"] = string(cfg[:boundary])
    file["trajectory_output_precision"] = string(writer.outtype)
    file["position_unit"] = "m"
    file["momentum_unit"] = "kg*m/s"
    write_injection_metadata!(file, cfg)
    close(file)
end

function output_paths(cfg, E)
    tag = string(Int(E)) * "_GeV"
    outdir = cfg[:output_dir]
    return (h5 = joinpath(outdir, "phase_space_" * tag * ".h5"), label = "E = " * string(Int(E)) * " GeV")
end

function run_gpu_ensemble(cfg, fields; energy_GeV)
    T = cfg[:precision]
    Bx, By, Bz, vx, vy, vz = fields
    nx, ny, nz = size(Bx)
    x, y, z = build_uniform_coords(cfg, nx, ny, nz, T)

    boundary = cfg[:boundary]
    if !(boundary == :open || boundary == :periodic)
        error("Unknown boundary mode: " * string(boundary) * ". Use :open or :periodic.")
    end
    periodic_boundary = boundary == :periodic

    trajectory_out_type = cfg[:trajectory_output_precision]

    B0 = reference_B0_T(Bx, By, Bz)
    gamma0 = energy_to_gamma(energy_GeV, T)
    Omega0 = T(reference_Omega0(B0, gamma0, Q_E, M_P))
    v0 = energy_to_speed(energy_GeV, T)
    trajectory_time = resolve_trajectory_time_grid(cfg, Omega0, v0, estimate_min_dx(x, y, z))
    save_time = resolve_save_stride(cfg, trajectory_time)
    trajectory_stride = save_time.trajectory_time_stride

    dt = T(trajectory_time.dt_s)
    nsteps = trajectory_time.n_integration_steps
    t_s = dt .* collect(T, 0:(nsteps - 1))
    t_norm = t_s .* Omega0
    t_gyroperiods = t_norm ./ T(2 * pi)
    save_indices = sampled_step_indices(nsteps, trajectory_stride)
    nsave = length(save_indices)

    x1, x2, x3, p1, p2, p3 = init_particles(cfg, x, y, z, gamma0, v0, T, Bx, By, Bz)
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

    alive_fraction = Vector{Float64}(undef, nsteps)
    phase_space_gib = bytes_to_gib(estimate_phase_space_bytes(cfg[:n_particles], nsave, trajectory_out_type))

    threads = 256
    blocks = cld(cfg[:n_particles], threads)

    println("  phase-space H5 = enabled")
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
    println("  output type    = ", trajectory_out_type)
    println("  est. size [GiB]= ", phase_space_gib)

    elapsed = NaN
    writer = nothing
    try
        writer = create_phase_space_writer(output_paths(cfg, energy_GeV).h5, cfg[:n_particles], nsave, trajectory_out_type)

        elapsed = @elapsed begin
            next_save_ptr = 1
            for si in 1:nsteps
                if (si - 1) % cfg[:progress_every] == 0
                    println("Energy ", energy_GeV, " GeV: step ", si, "/", nsteps)
                end

                do_push = si < nsteps
                @cuda threads=threads blocks=blocks advance_particles_kernel!(
                    dx1_step, dx2_step, dx3_step, dp1_step, dp2_step, dp3_step,
                    dx1, dx2, dx3, dp1, dp2, dp3, dalive,
                    dBx, dBy, dBz, dvx, dvy, dvz,
                    dt, xmin, ymin, zmin, xmax, ymax, zmax, dx, dy, dz, nx, ny, nz, do_push, periodic_boundary
                )

                alive_fraction[si] = Float64(sum(dalive)) / Float64(cfg[:n_particles])

                if next_save_ptr <= nsave && si == save_indices[next_save_ptr]
                    write_phase_space_step!(
                        writer,
                        next_save_ptr,
                        Array(dx1_step),
                        Array(dx2_step),
                        Array(dx3_step),
                        Array(dp1_step),
                        Array(dp2_step),
                        Array(dp3_step),
                    )
                    next_save_ptr += 1
                end
            end
            CUDA.synchronize()
        end

        finalize_phase_space_writer!(
            writer,
            cfg,
            energy_GeV,
            t_norm[save_indices],
            t_s[save_indices],
            t_gyroperiods[save_indices],
            alive_fraction[save_indices],
            trajectory_time,
            save_time,
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
        t_norm = Float64.(t_norm),
        t_s = Float64.(t_s),
        t_gyroperiods = Float64.(t_gyroperiods),
        alive_fraction = alive_fraction,
        nsave = nsave,
        phase_space_path = output_paths(cfg, energy_GeV).h5,
        phase_space_gib = phase_space_gib,
        elapsed = elapsed,
        Omega0 = Float64(Omega0),
        dt = Float64(dt),
        trajectory_time = trajectory_time,
        save_time = save_time,
        B0 = Float64(B0),
    )
end

function main()
    mkpath(CFG[:output_dir])
    fields = load_static_fields(CFG, CFG[:precision])

    println("GPU phase-space runner")
    println("  precision      = ", CFG[:precision])
    println("  particles      = ", CFG[:n_particles])
    println("  tOmega0_max    = ", CFG[:tOmega0_max])
    println("  boundary       = ", CFG[:boundary])
    println("  output_dir     = ", CFG[:output_dir])
    println("  field_subset   = ", CFG[:field_subset])

    for E in CFG[:energies]
        res = run_gpu_ensemble(CFG, fields; energy_GeV=E)
        println("Completed energy ", E, " GeV")
        println("  dt [s]         = ", res.dt)
        println("  Omega0 [1/s]   = ", res.Omega0)
        println("  steps          = ", length(res.t_norm))
        println("  saved steps    = ", res.nsave)
        println("  final alive    = ", res.alive_fraction[end])
        println("  elapsed [s]    = ", res.elapsed)
        println("  H5             = ", res.phase_space_path)
        println("  est. H5 [GiB]  = ", res.phase_space_gib)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
