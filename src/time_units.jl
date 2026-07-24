const REFERENCE_GYROPERIOD_NAME = "reference gyroperiod"
const TIME_REFERENCE_MODE = "total-field-mean-magnitude"
const TIME_REFERENCE_B0_DEFINITION = "B0_reference_T = mean(sqrt(Bx_total^2 + By_total^2 + Bz_total^2)) after magnetic-field conversion to tesla over the loaded reference grid"
const TIME_REFERENCE_ENERGY_DEFINITION = "Omega0_reference_s_inv = q_e * B0_reference_T / (gamma0 * m_p), where gamma0 is computed from the configured particle kinetic energy"

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

function campaign_time_reference(B0_reference_T::Real, Omega0_reference_s_inv::Real; source_mode::AbstractString="total", source_path::AbstractString="", source_identity::AbstractString=source_path, field_subset="nothing")
    Omega0 = Float64(Omega0_reference_s_inv)
    B0 = Float64(B0_reference_T)
    return (
        B0_reference_T = B0,
        Omega0_reference_s_inv = Omega0,
        reference_gyroperiod_s = reference_gyroperiod_s(Omega0),
        time_reference_mode = TIME_REFERENCE_MODE,
        time_reference_name = REFERENCE_GYROPERIOD_NAME,
        time_reference_definition = TIME_REFERENCE_B0_DEFINITION,
        time_reference_energy_definition = TIME_REFERENCE_ENERGY_DEFINITION,
        time_reference_source_mode = String(source_mode),
        time_reference_source_path = String(source_path),
        time_reference_source_identity = String(source_identity),
        time_reference_field_subset = string(field_subset),
    )
end

function require_explicit_campaign_time_reference(trajectory_mode, time_reference; caller::AbstractString="trajectory runner")
    time_reference !== nothing && return time_reference
    mode = lowercase(string(trajectory_mode))
    error("Trajectory mode '" * mode * "' requires an explicit campaign time reference derived from the total-field dataset. The " * caller * " must not derive B0_reference from the trajectory field. Use a total-field wrapper for standalone total-field runs.")
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

function validate_time_axes(t_s, t_norm, t_gyroperiods; key_name::AbstractString="time axes", require_uniform::Bool=false, rtol::Real=1.0e-10, atol::Real=1.0e-12)
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
    if require_uniform
        intervals = diff(Float64.(t_gyroperiods))
        save_interval = first(intervals)
        all(abs.(intervals .- save_interval) .<= Float64(atol) .+ Float64(rtol) .* abs(save_interval)) || error(nonuniform_time_axis_message(key_name, intervals, save_interval))
    end
    return n
end

function nonuniform_time_axis_message(key_name::AbstractString, intervals, expected_interval)
    diffs = abs.(Float64.(intervals) .- Float64(expected_interval))
    first_bad = findfirst(index -> diffs[index] > 1.0e-12 + 1.0e-10 * abs(Float64(expected_interval)), eachindex(diffs))
    first_bad === nothing && (first_bad = 1)
    return key_name * ": nonuniform primary time axis; fixed-index lag analysis is unsafe. Likely cause: an older cache appended an off-cadence final trajectory state. Required action: regenerate the trajectory cache with the current pipeline, then rerun transport analysis. Expected interval: $(expected_interval) reference gyroperiods. First irregular interval index: $(first_bad). Observed interval: $(intervals[first_bad]) reference gyroperiods."
end

function validate_time_axes_result(t_s, t_norm, t_gyroperiods; key_name::AbstractString="time axes", require_uniform::Bool=true, rtol::Real=1.0e-10, atol::Real=1.0e-12)
    n = length(t_s)
    length(t_norm) == n || return (valid=false, reason=:time_length_mismatch, message=key_name * ": t_norm length does not match t_s.")
    length(t_gyroperiods) == n || return (valid=false, reason=:time_length_mismatch, message=key_name * ": t_gyroperiods length does not match t_s.")
    n > 1 || return (valid=false, reason=:time_length_mismatch, message=key_name * ": need at least two time samples.")
    (all(isfinite, t_s) && all(isfinite, t_norm) && all(isfinite, t_gyroperiods)) || return (valid=false, reason=:inconsistent_time_units, message=key_name * ": time axes contain non-finite values.")
    (all(diff(Float64.(t_s)) .> 0.0) && all(diff(Float64.(t_norm)) .> 0.0) && all(diff(Float64.(t_gyroperiods)) .> 0.0)) || return (valid=false, reason=:inconsistent_time_units, message=key_name * ": time axes must be strictly increasing.")
    all(isapprox.(Float64.(t_norm) ./ (2.0 * pi), Float64.(t_gyroperiods); rtol=rtol, atol=atol)) || return (valid=false, reason=:inconsistent_time_units, message=key_name * ": t_norm and t_gyroperiods are inconsistent.")
    if require_uniform
        intervals = diff(Float64.(t_gyroperiods))
        save_interval = first(intervals)
        if !all(abs.(intervals .- save_interval) .<= Float64(atol) .+ Float64(rtol) .* abs(save_interval))
            return (valid=false, reason=:nonuniform_time_axis, message=nonuniform_time_axis_message(key_name, intervals, save_interval))
        end
    end
    return (valid=true, reason=:ok, message="ok")
end

function uniform_prefix_length(t_gyroperiods; rtol::Real=1.0e-10, atol::Real=1.0e-12)
    n = length(t_gyroperiods)
    n <= 2 && return n
    intervals = diff(Float64.(t_gyroperiods))
    expected = first(intervals)
    prefix = 1
    for interval in intervals
        if abs(interval - expected) <= Float64(atol) + Float64(rtol) * abs(expected)
            prefix += 1
        else
            break
        end
    end
    return prefix
end

function normalize_lag_boundary_policy(value)
    policy = Symbol(replace(lowercase(strip(String(value))), "-" => "_"))
    policy in (:strict, :nearest) || error("lag_boundary_policy must be strict or nearest.")
    return policy
end

function normalize_lag_range_policy(value)
    policy = Symbol(replace(lowercase(strip(String(value))), "-" => "_"))
    policy in (:fixed, :first_cache_step, :common_cache_intersection) || error("lag_range_policy must be fixed, first-cache-step, or common-cache-intersection.")
    return policy
end

function normalize_lag_common_scope(value)
    scope = Symbol(replace(lowercase(strip(String(value))), "-" => "_"))
    scope in (:campaign, :reference_group) || error("lag_common_scope must be campaign or reference-group.")
    return scope
end

function lag_cache_summary(t_gyroperiods)
    nsteps = length(t_gyroperiods)
    nsteps > 1 || error("Need at least two cached samples to resolve lags.")
    offsets = Float64.(t_gyroperiods[2:end]) .- Float64(t_gyroperiods[1])
    cache_min_gp = first(offsets)
    cache_max_gp = last(offsets)
    cache_save_interval_gp = length(offsets) == 1 ? cache_min_gp : first(diff(Float64.(t_gyroperiods)))
    tolerance = 1.0e-12 + 1.0e-10 * max(abs(cache_save_interval_gp), abs(cache_max_gp), 1.0)
    return (offsets=offsets, cache_min_gp=cache_min_gp, cache_max_gp=cache_max_gp, cache_save_interval_gp=cache_save_interval_gp, tolerance=tolerance)
end

function nearest_positive_cache_lag(requested_gp::Real, cache)
    requested = Float64(requested_gp)
    isfinite(requested) && requested > 0.0 || error("Lag boundary values must be finite and positive reference gyroperiods.")
    nearest_local = argmin(abs.(cache.offsets .- requested))
    nearest_step = Int(nearest_local)
    nearest_step >= 1 || error("Lag boundary cannot resolve to zero lag.")
    return nearest_step, Float64(cache.offsets[nearest_local]), abs(Float64(cache.offsets[nearest_local]) - requested)
end

function resolve_one_lag_boundary(requested_gp::Real, cache, cfg, boundary_name::AbstractString; analysis_label::AbstractString="lag")
    policy = normalize_lag_boundary_policy(get(cfg, :lag_boundary_policy, :strict))
    requested = Float64(requested_gp)
    nearest_step, actual, abs_error = nearest_positive_cache_lag(requested, cache)
    if policy == :strict
        if boundary_name == "minimum"
            requested >= cache.cache_min_gp - cache.tolerance || error("Requested lag range minimum $(requested) reference gyroperiods is outside representable cache range [$(cache.cache_min_gp), $(cache.cache_max_gp)] reference gyroperiods. Use lag_boundary_policy = \"nearest\" with an explicit tolerance for discretization-scale cache-cadence mismatches, use lag_range_policy = \"first-cache-step\", or adjust lag_min_gyroperiods.")
        else
            requested <= cache.cache_max_gp + cache.tolerance || error("Requested lag range maximum $(requested) reference gyroperiods is outside representable cache range [$(cache.cache_min_gp), $(cache.cache_max_gp)] reference gyroperiods. Use lag_boundary_policy = \"nearest\" with an explicit tolerance for discretization-scale cache-cadence mismatches or adjust lag_max_gyroperiods/trajectory_duration_gyroperiods.")
        end
        return requested
    end

    max_relative_error = Float64(get(cfg, :max_lag_boundary_relative_error, 0.0))
    isfinite(max_relative_error) && max_relative_error >= 0.0 || error("max_lag_boundary_relative_error must be finite and nonnegative.")
    abs_error <= 0.5 * cache.cache_save_interval_gp + cache.tolerance || error("Requested lag boundary $(requested) reference gyroperiods is too far from the nearest cached lag $(actual) reference gyroperiods for nearest policy. Absolute error $(abs_error) exceeds half the cache interval $(0.5 * cache.cache_save_interval_gp).")
    rel_error = abs_error / max(abs(requested), eps(Float64))
    rel_error <= max_relative_error + cache.tolerance || error("Requested lag boundary $(requested) reference gyroperiods maps to $(actual) reference gyroperiods with relative error $(rel_error), exceeding max_lag_boundary_relative_error=$(max_relative_error).")
    if abs_error > cache.tolerance
        @warn string(analysis_label, " lag boundary adjusted by nearest policy") boundary=boundary_name requested_gyroperiods=requested resolved_gyroperiods=actual absolute_error_gyroperiods=abs_error relative_error=rel_error cache_interval_gyroperiods=cache.cache_save_interval_gp configured_tolerance=max_relative_error
    end
    return actual
end

function configured_lag_bounds(cfg, cache)
    configured_min = get(cfg, :lag_min_gyroperiods, nothing)
    configured_max = get(cfg, :lag_max_gyroperiods, nothing)
    range_policy = normalize_lag_range_policy(get(cfg, :lag_range_policy, :fixed))

    if range_policy == :first_cache_step
        configured_min === nothing || error("lag_range_policy = first-cache-step requires lag_min_gyroperiods to be absent or nothing.")
        lag_min = cache.cache_min_gp
        lag_max = configured_max === nothing ? cache.cache_max_gp : assert_finite_positive(configured_max, "lag_max_gyroperiods")
        return configured_min, configured_max, lag_min, lag_max, NaN, NaN
    end

    lag_min = configured_min === nothing ? cache.cache_min_gp : assert_finite_positive(configured_min, "lag_min_gyroperiods")
    lag_max = configured_max === nothing ? cache.cache_max_gp : assert_finite_positive(configured_max, "lag_max_gyroperiods")
    if range_policy == :common_cache_intersection
        common_min = Float64(get(cfg, :common_cache_lag_min_gyroperiods, NaN))
        common_max = Float64(get(cfg, :common_cache_lag_max_gyroperiods, NaN))
        isfinite(common_min) && isfinite(common_max) || error("lag_range_policy = common-cache-intersection requires a preflight common cache range.")
        common_max > common_min || error("Common cache lag intersection is empty.")
        lag_min = max(lag_min, common_min)
        lag_max = min(lag_max, common_max)
        lag_max > lag_min || error("Effective lag range is empty after applying common-cache-intersection.")
        return configured_min, configured_max, lag_min, lag_max, common_min, common_max
    end
    return configured_min, configured_max, lag_min, lag_max, NaN, NaN
end

function build_requested_lags_from_bounds(cfg, lag_min::Real, lag_max::Real)
    mode = cfg[:lag_mode]
    lag_min_f = Float64(lag_min)
    lag_max_f = Float64(lag_max)
    lag_max_f >= lag_min_f || error("lag_max_gyroperiods must be >= lag_min_gyroperiods after policy resolution.")
    if mode == :uniform_samples
        n_lag_samples = Int(cfg[:n_lag_samples])
        n_lag_samples > 0 || error("n_lag_samples must be positive.")
        return collect(range(lag_min_f, lag_max_f, length=n_lag_samples)), :gyroperiods
    elseif mode == :stride
        lag_stride = assert_finite_positive(cfg[:lag_stride_gyroperiods], "lag_stride_gyroperiods")
        return collect(lag_min_f:lag_stride:(lag_max_f + lag_stride * 1.0e-12)), :gyroperiods
    end
    error("Unknown lag_mode: $(mode)")
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

function resolve_lag_grid(cfg, t_gyroperiods; min_unique_lags::Integer=2)
    nsteps = length(t_gyroperiods)
    nsteps > 1 || error("Need at least two cached samples to resolve lags.")
    max_cache_lag_step = nsteps - 1
    cache = lag_cache_summary(t_gyroperiods)
    range_policy = normalize_lag_range_policy(get(cfg, :lag_range_policy, :fixed))
    boundary_policy = normalize_lag_boundary_policy(get(cfg, :lag_boundary_policy, :strict))
    boundary_relative_error = Float64(get(cfg, :max_lag_boundary_relative_error, 0.0))
    isfinite(boundary_relative_error) && boundary_relative_error >= 0.0 || error("max_lag_boundary_relative_error must be finite and nonnegative.")
    analysis_label = String(get(cfg, :analysis_label, "transport"))

    configured_min_gp = get(cfg, :lag_min_gyroperiods, nothing)
    configured_max_gp = get(cfg, :lag_max_gyroperiods, nothing)
    common_min_gp = NaN
    common_max_gp = NaN
    effective_min_gp = NaN
    effective_max_gp = NaN

    requested_tau_gp, source = if range_policy == :fixed && configured_min_gp === nothing
        requested_lag_grid_gyroperiods(cfg, max_cache_lag_step, t_gyroperiods)
    else
        configured_min_gp, configured_max_gp, requested_min_gp, requested_max_gp, common_min_gp, common_max_gp = configured_lag_bounds(cfg, cache)
        if range_policy == :common_cache_intersection
            effective_min_gp = requested_min_gp
            effective_max_gp = requested_max_gp
        else
            effective_min_gp = resolve_one_lag_boundary(requested_min_gp, cache, cfg, "minimum"; analysis_label=analysis_label)
            effective_max_gp = resolve_one_lag_boundary(requested_max_gp, cache, cfg, "maximum"; analysis_label=analysis_label)
        end
        effective_max_gp > effective_min_gp || error("Effective lag range resolves to zero width after applying lag boundary policy.")
        build_requested_lags_from_bounds(cfg, effective_min_gp, effective_max_gp)
    end
    isempty(requested_tau_gp) && error("No requested lag values were generated.")

    offsets = cache.offsets
    cache_min_gp = cache.cache_min_gp
    cache_max_gp = cache.cache_max_gp
    cache_save_interval_gp = cache.cache_save_interval_gp
    requested_min_gp = minimum(Float64.(requested_tau_gp))
    requested_max_gp = maximum(Float64.(requested_tau_gp))
    tolerance = cache.tolerance
    if boundary_policy == :strict && (requested_min_gp < cache_min_gp - tolerance || requested_max_gp > cache_max_gp + tolerance)
        error("Requested lag range [$(requested_min_gp), $(requested_max_gp)] reference gyroperiods is outside representable cache range [$(cache_min_gp), $(cache_max_gp)] reference gyroperiods. Cache save interval is $(cache_save_interval_gp) reference gyroperiods and cache duration is $(cache_max_gp) reference gyroperiods. Adjust lag_min_gyroperiods/lag_max_gyroperiods or regenerate the cache with a different trajectory_save_interval_gyroperiods or trajectory_duration_gyroperiods.")
    end

    lag_steps = Int[]
    requested_unique_gp = Float64[]
    actual_gp = Float64[]
    errors_gp = Float64[]
    seen = Set{Int}()
    for requested_gp in requested_tau_gp
        isfinite(requested_gp) && requested_gp > 0.0 || error("Requested lag values must be finite and positive reference gyroperiods.")
        nearest_local = argmin(abs.(offsets .- requested_gp))
        lag_step = Int(nearest_local)
        1 <= lag_step <= max_cache_lag_step || error("Resolved lag step is outside the cached trajectory.")
        mapping_error = abs(Float64(offsets[nearest_local]) - Float64(requested_gp))
        mapping_error <= 0.5 * cache_save_interval_gp + tolerance || error("Internal lag mapping inconsistency: requested $(requested_gp) reference gyroperiods mapped to $(offsets[nearest_local]), exceeding half the cache interval $(cache_save_interval_gp / 2).")
        if !(lag_step in seen)
            push!(seen, lag_step)
            push!(lag_steps, lag_step)
            push!(requested_unique_gp, Float64(requested_gp))
            actual = Float64(t_gyroperiods[lag_step + 1] - t_gyroperiods[1])
            push!(actual_gp, actual)
            push!(errors_gp, mapping_error)
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
        common_requested_tau_gyroperiods = Float64.(requested_tau_gp),
        tau_gyroperiods = actual_gp,
        tau_norm = tau_norm,
        lag_mapping_error_gyroperiods = errors_gp,
        max_lag_mapping_error_gyroperiods = isempty(errors_gp) ? 0.0 : maximum(errors_gp),
        max_lag_mapping_relative_error = isempty(errors_gp) ? 0.0 : maximum(errors_gp ./ max.(abs.(requested_unique_gp), eps(Float64))),
        requested_lag_count = length(requested_tau_gp),
        unique_lag_count = length(lag_steps),
        duplicate_lag_mapping_count = length(requested_tau_gp) - length(lag_steps),
        duplicate_lag_mapping_fraction = isempty(requested_tau_gp) ? 0.0 : (length(requested_tau_gp) - length(lag_steps)) / length(requested_tau_gp),
        cache_lag_min_gyroperiods = cache_min_gp,
        cache_lag_max_gyroperiods = cache_max_gp,
        cache_save_interval_gyroperiods = cache_save_interval_gp,
        lag_range_policy = range_policy,
        lag_boundary_policy = boundary_policy,
        max_lag_boundary_relative_error = boundary_relative_error,
        configured_lag_min_gyroperiods = configured_min_gp === nothing ? NaN : Float64(configured_min_gp),
        configured_lag_max_gyroperiods = configured_max_gp === nothing ? NaN : Float64(configured_max_gp),
        common_cache_lag_min_gyroperiods = common_min_gp,
        common_cache_lag_max_gyroperiods = common_max_gp,
        effective_lag_min_gyroperiods = isfinite(effective_min_gp) ? effective_min_gp : requested_min_gp,
        effective_lag_max_gyroperiods = isfinite(effective_max_gp) ? effective_max_gp : requested_max_gp,
        lag_comparison_group_identity = String(get(cfg, :lag_comparison_group_identity, "not-applicable")),
        lag_common_scope = normalize_lag_common_scope(get(cfg, :lag_common_scope, :campaign)),
        preflight_job_id = String(get(cfg, :preflight_job_id, "not-applicable")),
        preflight_reference_group_id = String(get(cfg, :preflight_reference_group_id, "not-applicable")),
        lag_group_member_count = Int(get(cfg, :lag_group_member_count, 1)),
        lag_group_member_modes = String(get(cfg, :lag_group_member_modes, "not-applicable")),
        lag_group_member_energies_GeV = String(get(cfg, :lag_group_member_energies_GeV, "not-applicable")),
        lag_source = source,
    )
end

function write_time_reference_metadata!(file, time_reference)
    file["Omega0"] = Float64[time_reference.Omega0_reference_s_inv]
    file["B0_T"] = Float64[time_reference.B0_reference_T]
    file["Omega0_reference_s_inv"] = Float64[time_reference.Omega0_reference_s_inv]
    file["B0_reference_T"] = Float64[time_reference.B0_reference_T]
    file["reference_gyroperiod_s"] = Float64[time_reference.reference_gyroperiod_s]
    file["time_reference_name"] = REFERENCE_GYROPERIOD_NAME
    file["time_reference_mode"] = time_reference.time_reference_mode
    file["time_reference_definition"] = time_reference.time_reference_definition
    file["time_reference_B0_definition"] = time_reference.time_reference_definition
    file["time_reference_energy_definition"] = time_reference.time_reference_energy_definition
    file["time_reference_source_mode"] = time_reference.time_reference_source_mode
    file["time_reference_source_path"] = time_reference.time_reference_source_path
    file["time_reference_source_identity"] = time_reference.time_reference_source_identity
    file["time_reference_field_subset"] = time_reference.time_reference_field_subset
    return nothing
end

function write_trajectory_time_metadata!(file, trajectory_time, save_time)
    file["requested_trajectory_duration_gyroperiods"] = Float64[trajectory_time.requested_trajectory_duration_gyroperiods]
    file["actual_trajectory_duration_gyroperiods"] = Float64[trajectory_time.actual_trajectory_duration_gyroperiods]
    file["actual_integration_duration_gyroperiods"] = Float64[trajectory_time.actual_trajectory_duration_gyroperiods]
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
