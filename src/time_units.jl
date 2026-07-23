const REFERENCE_GYROPERIOD_NAME = "reference gyroperiod"
const TIME_REFERENCE_B0_DEFINITION = "B0_T = mean(sqrt(Bx^2 + By^2 + Bz^2)) after magnetic-field conversion to tesla over the loaded trajectory field grid"
const TIME_REFERENCE_ENERGY_DEFINITION = "Omega0 = q_e * B0_T / (gamma0 * m_p), where gamma0 is computed from the configured particle kinetic energy"

reference_gyroperiod_s(Omega0::Real) = 2.0 * pi / Float64(Omega0)
gyroperiods_to_tOmega0(n_gyroperiods::Real) = 2.0 * pi * Float64(n_gyroperiods)
tOmega0_to_gyroperiods(tOmega0::Real) = Float64(tOmega0) / (2.0 * pi)
gyroperiods_to_seconds(n_gyroperiods::Real, Omega0::Real) = gyroperiods_to_tOmega0(n_gyroperiods) / Float64(Omega0)
seconds_to_gyroperiods(t_s::Real, Omega0::Real) = Float64(t_s) * Float64(Omega0) / (2.0 * pi)

function assert_finite_positive(value::Real, key_name::AbstractString)
    finite_value = Float64(value)
    isfinite(finite_value) && finite_value > 0.0 || error(key_name * " must be finite and positive.")
    return finite_value
end

function reference_B0_T(Bx, By, Bz)
    return mean(sqrt.(Bx .^ 2 .+ By .^ 2 .+ Bz .^ 2))
end

function reference_Omega0(B0_T::Real, gamma0::Real, charge_C::Real, mass_kg::Real)
    return Float64(charge_C) * Float64(B0_T) / (Float64(gamma0) * Float64(mass_kg))
end

function resolve_eta_requested(cfg)
    if haskey(cfg, :integration_steps_per_gyroperiod) && cfg[:integration_steps_per_gyroperiod] !== nothing
        steps_per_gyroperiod = assert_finite_positive(cfg[:integration_steps_per_gyroperiod], "integration_steps_per_gyroperiod")
        return 2.0 * pi / steps_per_gyroperiod, steps_per_gyroperiod, :gyroperiods
    end
    eta = assert_finite_positive(cfg[:eta], "eta")
    return eta, 2.0 * pi / eta, :legacy_eta
end

function requested_duration_gyroperiods(cfg)
    if haskey(cfg, :trajectory_duration_gyroperiods) && cfg[:trajectory_duration_gyroperiods] !== nothing
        return assert_finite_positive(cfg[:trajectory_duration_gyroperiods], "trajectory_duration_gyroperiods"), :gyroperiods
    end
    tOmega0_max = assert_finite_positive(cfg[:tOmega0_max], "tOmega0_max")
    return tOmega0_to_gyroperiods(tOmega0_max), :legacy_tOmega0
end

function resolve_trajectory_time_grid(cfg, Omega0::Real, v0::Real, min_dx::Real)
    eta_requested, requested_steps_per_gyroperiod, integration_source = resolve_eta_requested(cfg)
    dt_gyro_s = eta_requested / Float64(Omega0)
    dt_cfl_s = Inf
    timestep_limited_by = "gyro"
    dt_s = dt_gyro_s
    if Bool(get(cfg, :use_cfl, false))
        cfl = assert_finite_positive(cfg[:cfl], "cfl")
        dt_cfl_s = cfl * Float64(min_dx) / Float64(v0)
        if dt_cfl_s < dt_s
            dt_s = dt_cfl_s
            timestep_limited_by = "cfl"
        end
    end

    requested_duration_gp, duration_source = requested_duration_gyroperiods(cfg)
    requested_duration_s = gyroperiods_to_seconds(requested_duration_gp, Omega0)
    n_integration_steps = Int(floor(requested_duration_s / dt_s)) + 1
    n_integration_steps > 1 || error("Trajectory duration resolves to fewer than two integration samples.")
    actual_duration_s = (n_integration_steps - 1) * dt_s
    actual_duration_tOmega0 = actual_duration_s * Float64(Omega0)
    actual_duration_gp = tOmega0_to_gyroperiods(actual_duration_tOmega0)
    actual_steps_per_gyroperiod = reference_gyroperiod_s(Omega0) / dt_s

    return (
        eta_requested = eta_requested,
        integration_source = integration_source,
        requested_integration_steps_per_gyroperiod = requested_steps_per_gyroperiod,
        actual_integration_steps_per_gyroperiod = actual_steps_per_gyroperiod,
        dt_s = dt_s,
        dt_tOmega0 = dt_s * Float64(Omega0),
        dt_gyroperiods = seconds_to_gyroperiods(dt_s, Omega0),
        dt_gyro_s = dt_gyro_s,
        dt_cfl_s = dt_cfl_s,
        timestep_limited_by = timestep_limited_by,
        duration_source = duration_source,
        requested_trajectory_duration_gyroperiods = requested_duration_gp,
        requested_trajectory_duration_s = requested_duration_s,
        actual_trajectory_duration_gyroperiods = actual_duration_gp,
        actual_trajectory_duration_tOmega0 = actual_duration_tOmega0,
        actual_trajectory_duration_s = actual_duration_s,
        n_integration_steps = n_integration_steps,
        rounding_policy = "floor(requested_duration_s / dt_s) + 1 integration samples",
    )
end

function resolve_save_stride(cfg, trajectory_time)
    if haskey(cfg, :trajectory_save_interval_gyroperiods) && cfg[:trajectory_save_interval_gyroperiods] !== nothing
        requested_interval_gp = assert_finite_positive(cfg[:trajectory_save_interval_gyroperiods], "trajectory_save_interval_gyroperiods")
        stride = max(1, round(Int, requested_interval_gp / trajectory_time.dt_gyroperiods))
        actual_interval_gp = stride * trajectory_time.dt_gyroperiods
        return (
            trajectory_time_stride = stride,
            save_interval_source = :gyroperiods,
            requested_trajectory_save_interval_gyroperiods = requested_interval_gp,
            actual_trajectory_save_interval_gyroperiods = actual_interval_gp,
            requested_trajectory_save_interval_s = requested_interval_gp * trajectory_time.dt_s / trajectory_time.dt_gyroperiods,
            actual_trajectory_save_interval_s = stride * trajectory_time.dt_s,
            rounding_policy = "round(requested_save_interval_s / dt_s) clamped to at least one integration step",
        )
    end
    stride = Int(cfg[:trajectory_time_stride])
    stride >= 1 || error("trajectory_time_stride must be >= 1")
    actual_interval_gp = stride * trajectory_time.dt_gyroperiods
    return (
        trajectory_time_stride = stride,
        save_interval_source = :legacy_steps,
        requested_trajectory_save_interval_gyroperiods = actual_interval_gp,
        actual_trajectory_save_interval_gyroperiods = actual_interval_gp,
        requested_trajectory_save_interval_s = stride * trajectory_time.dt_s,
        actual_trajectory_save_interval_s = stride * trajectory_time.dt_s,
        rounding_policy = "legacy integer trajectory_time_stride",
    )
end

function t_gyroperiods_from_axes(t_s, t_norm, Omega0=nothing)
    if t_norm !== nothing
        return Float64.(t_norm) ./ (2.0 * pi)
    elseif Omega0 !== nothing
        return Float64.(t_s) .* Float64(Omega0) ./ (2.0 * pi)
    end
    error("Cannot resolve t_gyroperiods without t_norm or Omega0.")
end

function validate_time_axes(t_s, t_norm, t_gyroperiods; key_name::AbstractString="time axes")
    n = length(t_s)
    length(t_norm) == n || error(key_name * ": t_norm length does not match t_s.")
    length(t_gyroperiods) == n || error(key_name * ": t_gyroperiods length does not match t_s.")
    n > 1 || error(key_name * ": need at least two time samples.")
    all(isfinite, t_s) || error(key_name * ": t_s contains non-finite values.")
    all(isfinite, t_norm) || error(key_name * ": t_norm contains non-finite values.")
    all(isfinite, t_gyroperiods) || error(key_name * ": t_gyroperiods contains non-finite values.")
    all(diff(Float64.(t_s)) .> 0.0) || error(key_name * ": t_s must be strictly increasing.")
    all(diff(Float64.(t_norm)) .> 0.0) || error(key_name * ": t_norm must be strictly increasing.")
    all(diff(Float64.(t_gyroperiods)) .> 0.0) || error(key_name * ": t_gyroperiods must be strictly increasing.")
    all(isapprox.(Float64.(t_norm) ./ (2.0 * pi), Float64.(t_gyroperiods); rtol=1.0e-10, atol=1.0e-12)) || error(key_name * ": t_norm and t_gyroperiods are inconsistent.")
    return n
end

function requested_lag_grid_gyroperiods(cfg, max_cache_lag_step::Integer, t_gyroperiods)
    mode = cfg[:lag_mode]
    if haskey(cfg, :lag_min_gyroperiods) && cfg[:lag_min_gyroperiods] !== nothing
        lag_min = assert_finite_positive(cfg[:lag_min_gyroperiods], "lag_min_gyroperiods")
        lag_max = haskey(cfg, :lag_max_gyroperiods) && cfg[:lag_max_gyroperiods] !== nothing ?
            assert_finite_positive(cfg[:lag_max_gyroperiods], "lag_max_gyroperiods") :
            Float64(t_gyroperiods[max_cache_lag_step + 1] - t_gyroperiods[1])
        lag_max >= lag_min || error("lag_max_gyroperiods must be >= lag_min_gyroperiods.")
        if mode == :uniform_samples
            n_lag_samples = Int(cfg[:n_lag_samples])
            n_lag_samples > 0 || error("n_lag_samples must be positive.")
            return collect(range(lag_min, lag_max, length=n_lag_samples)), :gyroperiods
        elseif mode == :stride
            lag_stride = assert_finite_positive(cfg[:lag_stride_gyroperiods], "lag_stride_gyroperiods")
            return collect(lag_min:lag_stride:(lag_max + lag_stride * 1.0e-12)), :gyroperiods
        end
        error("Unknown lag_mode: $(mode)")
    end

    min_lag_step = max(1, Int(get(cfg, :min_lag_steps, 1)))
    max_lag_step = cfg[:max_lag_steps] === nothing ? max_cache_lag_step : min(Int(cfg[:max_lag_steps]), max_cache_lag_step)
    max_lag_step >= min_lag_step || error("No lag steps selected. Check min_lag_steps, max_lag_steps, and saved trajectory length.")
    legacy_steps = if mode == :uniform_samples
        n_lags = min(Int(cfg[:n_lag_samples]), max_lag_step - min_lag_step + 1)
        unique(round.(Int, range(min_lag_step, max_lag_step, length=n_lags)))
    elseif mode == :stride
        stride = Int(cfg[:lag_step_stride])
        stride >= 1 || error("lag_step_stride must be >= 1")
        collect(min_lag_step:stride:max_lag_step)
    else
        error("Unknown lag_mode: $(mode)")
    end
    return [Float64(t_gyroperiods[step + 1] - t_gyroperiods[1]) for step in legacy_steps], :legacy_steps
end

function resolve_lag_grid(cfg, t_gyroperiods; min_unique_lags::Integer=1)
    nsteps = length(t_gyroperiods)
    nsteps > 1 || error("Need at least two cached samples to resolve lags.")
    max_cache_lag_step = nsteps - 1
    requested_tau_gp, source = requested_lag_grid_gyroperiods(cfg, max_cache_lag_step, t_gyroperiods)
    isempty(requested_tau_gp) && error("No requested lag values were generated.")

    lag_steps = Int[]
    requested_unique_gp = Float64[]
    actual_gp = Float64[]
    errors_gp = Float64[]
    seen = Set{Int}()
    for requested_gp in requested_tau_gp
        isfinite(requested_gp) && requested_gp > 0.0 || error("Requested lag values must be finite and positive reference gyroperiods.")
        offsets = Float64.(t_gyroperiods[2:end]) .- Float64(t_gyroperiods[1])
        nearest_local = argmin(abs.(offsets .- requested_gp))
        lag_step = Int(nearest_local)
        1 <= lag_step <= max_cache_lag_step || error("Resolved lag step is outside the cached trajectory.")
        if !(lag_step in seen)
            push!(seen, lag_step)
            push!(lag_steps, lag_step)
            push!(requested_unique_gp, Float64(requested_gp))
            actual = Float64(t_gyroperiods[lag_step + 1] - t_gyroperiods[1])
            push!(actual_gp, actual)
            push!(errors_gp, abs(actual - Float64(requested_gp)))
        end
    end
    length(lag_steps) >= min_unique_lags || error("Lag grid resolved to fewer than $(min_unique_lags) unique positive cached-step offsets.")
    order = sortperm(lag_steps)
    lag_steps = lag_steps[order]
    requested_unique_gp = requested_unique_gp[order]
    actual_gp = actual_gp[order]
    errors_gp = errors_gp[order]
    tau_norm = gyroperiods_to_tOmega0.(actual_gp)
    return (
        lag_steps = lag_steps,
        requested_tau_gyroperiods = requested_unique_gp,
        tau_gyroperiods = actual_gp,
        tau_norm = tau_norm,
        lag_mapping_error_gyroperiods = errors_gp,
        max_lag_mapping_error_gyroperiods = isempty(errors_gp) ? 0.0 : maximum(errors_gp),
        lag_source = source,
    )
end

function write_time_reference_metadata!(file, Omega0::Real, B0_T::Real)
    file["Omega0"] = Float64[Omega0]
    file["B0_T"] = Float64[B0_T]
    file["reference_gyroperiod_s"] = Float64[reference_gyroperiod_s(Omega0)]
    file["time_reference_name"] = REFERENCE_GYROPERIOD_NAME
    file["time_reference_B0_definition"] = TIME_REFERENCE_B0_DEFINITION
    file["time_reference_energy_definition"] = TIME_REFERENCE_ENERGY_DEFINITION
    return nothing
end

function write_trajectory_time_metadata!(file, trajectory_time, save_time)
    file["requested_trajectory_duration_gyroperiods"] = Float64[trajectory_time.requested_trajectory_duration_gyroperiods]
    file["actual_trajectory_duration_gyroperiods"] = Float64[trajectory_time.actual_trajectory_duration_gyroperiods]
    file["actual_trajectory_duration_tOmega0"] = Float64[trajectory_time.actual_trajectory_duration_tOmega0]
    file["actual_trajectory_duration_s"] = Float64[trajectory_time.actual_trajectory_duration_s]
    file["requested_integration_steps_per_gyroperiod"] = Float64[trajectory_time.requested_integration_steps_per_gyroperiod]
    file["actual_integration_steps_per_gyroperiod"] = Float64[trajectory_time.actual_integration_steps_per_gyroperiod]
    file["dt_s"] = Float64[trajectory_time.dt_s]
    file["dt_tOmega0"] = Float64[trajectory_time.dt_tOmega0]
    file["dt_gyroperiods"] = Float64[trajectory_time.dt_gyroperiods]
    file["timestep_limited_by"] = trajectory_time.timestep_limited_by
    file["requested_trajectory_save_interval_gyroperiods"] = Float64[save_time.requested_trajectory_save_interval_gyroperiods]
    file["actual_trajectory_save_interval_gyroperiods"] = Float64[save_time.actual_trajectory_save_interval_gyroperiods]
    file["trajectory_time_stride"] = Int[save_time.trajectory_time_stride]
    return nothing
end
