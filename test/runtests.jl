using Test
using Statistics

include(joinpath(@__DIR__, "..", "src", "time_units.jl"))

@testset "reference gyroperiod conversions" begin
    Omega0 = 4.0
    @test reference_gyroperiod_s(Omega0) ≈ pi / 2.0
    @test gyroperiods_to_tOmega0(1.0) ≈ 2.0 * pi
    @test tOmega0_to_gyroperiods(2.0 * pi) ≈ 1.0
    @test gyroperiods_to_seconds(3.0, Omega0) ≈ 3.0 * 2.0 * pi / Omega0
    @test seconds_to_gyroperiods(gyroperiods_to_seconds(2.5, Omega0), Omega0) ≈ 2.5
end

@testset "trajectory time resolution" begin
    cfg = Dict{Symbol, Any}(
        :integration_steps_per_gyroperiod => 64.0,
        :eta => 0.1,
        :trajectory_duration_gyroperiods => 10.0,
        :tOmega0_max => 100.0,
        :use_cfl => false,
        :cfl => 0.2,
        :trajectory_save_interval_gyroperiods => 0.25,
        :trajectory_time_stride => 1,
    )
    resolved = resolve_trajectory_time_grid(cfg, 2.0, 1.0, 1.0)
    @test resolved.eta_requested ≈ 2.0 * pi / 64.0
    @test resolved.requested_integration_steps_per_gyroperiod ≈ 64.0
    @test resolved.actual_integration_steps_per_gyroperiod ≈ 64.0
    @test resolved.timestep_limited_by == "gyro"
    @test resolved.actual_trajectory_duration_gyroperiods <= 10.0

    save_time = resolve_save_stride(cfg, resolved)
    @test save_time.trajectory_time_stride == 16
    @test save_time.actual_trajectory_save_interval_gyroperiods ≈ 0.25

    cfl_cfg = copy(cfg)
    cfl_cfg[:use_cfl] = true
    cfl_resolved = resolve_trajectory_time_grid(cfl_cfg, 2.0, 100.0, 1.0)
    @test cfl_resolved.timestep_limited_by == "cfl"
    @test cfl_resolved.actual_integration_steps_per_gyroperiod > cfl_resolved.requested_integration_steps_per_gyroperiod
end

@testset "legacy equivalence" begin
    legacy_cfg = Dict{Symbol, Any}(
        :integration_steps_per_gyroperiod => nothing,
        :eta => 0.1,
        :trajectory_duration_gyroperiods => nothing,
        :tOmega0_max => 20.0,
        :use_cfl => false,
        :cfl => 0.2,
        :trajectory_save_interval_gyroperiods => nothing,
        :trajectory_time_stride => 3,
    )
    resolved = resolve_trajectory_time_grid(legacy_cfg, 5.0, 1.0, 1.0)
    @test resolved.eta_requested ≈ 0.1
    @test resolved.requested_trajectory_duration_gyroperiods ≈ 20.0 / (2.0 * pi)
    @test resolved.n_integration_steps == Int(floor((20.0 / 5.0) / (0.1 / 5.0))) + 1
    save_time = resolve_save_stride(legacy_cfg, resolved)
    @test save_time.trajectory_time_stride == 3
end

@testset "lag mapping" begin
    t_gp = collect(0.0:0.25:5.0)
    cfg = Dict{Symbol, Any}(
        :lag_mode => :uniform_samples,
        :lag_range_policy => :fixed,
        :lag_boundary_policy => :strict,
        :max_lag_boundary_relative_error => 0.0,
        :lag_min_gyroperiods => 0.25,
        :lag_max_gyroperiods => 1.1,
        :lag_stride_gyroperiods => nothing,
        :min_lag_steps => 1,
        :max_lag_steps => nothing,
        :lag_step_stride => 1,
        :n_lag_samples => 5,
    )
    lag_grid = resolve_lag_grid(cfg, t_gp)
    @test all(lag_grid.lag_steps .>= 1)
    @test issorted(lag_grid.lag_steps)
    @test length(unique(lag_grid.lag_steps)) == length(lag_grid.lag_steps)
    @test all(lag_grid.tau_gyroperiods .> 0.0)
    @test lag_grid.tau_norm ≈ 2.0 * pi .* lag_grid.tau_gyroperiods

    stride_cfg = copy(cfg)
    stride_cfg[:lag_mode] = :stride
    stride_cfg[:lag_min_gyroperiods] = 0.25
    stride_cfg[:lag_max_gyroperiods] = 1.0
    stride_cfg[:lag_stride_gyroperiods] = 0.25
    stride_grid = resolve_lag_grid(stride_cfg, t_gp)
    @test stride_grid.lag_steps == [1, 2, 3, 4]
    @test stride_grid.tau_gyroperiods ≈ [0.25, 0.5, 0.75, 1.0]

    too_small = copy(cfg)
    too_small[:lag_min_gyroperiods] = 0.01
    too_small[:lag_max_gyroperiods] = 0.5
    @test_throws ErrorException resolve_lag_grid(too_small, t_gp)

    too_large = copy(cfg)
    too_large[:lag_min_gyroperiods] = 0.5
    too_large[:lag_max_gyroperiods] = 8.0
    @test_throws ErrorException resolve_lag_grid(too_large, t_gp)

    duplicate_cfg = copy(cfg)
    duplicate_cfg[:lag_min_gyroperiods] = 0.26
    duplicate_cfg[:lag_max_gyroperiods] = 0.49
    duplicate_cfg[:n_lag_samples] = 4
    duplicate_grid = resolve_lag_grid(duplicate_cfg, t_gp; min_unique_lags=1)
    @test duplicate_grid.requested_lag_count == 4
    @test duplicate_grid.unique_lag_count < duplicate_grid.requested_lag_count
    @test duplicate_grid.duplicate_lag_mapping_count > 0
    @test duplicate_grid.duplicate_lag_mapping_fraction > 0.0
end

@testset "lag boundary policies" begin
    t_gp = collect(0.0:0.016087419809019843:5.984520168955382)
    cfg = Dict{Symbol, Any}(
        :lag_mode => :uniform_samples,
        :lag_range_policy => :fixed,
        :lag_boundary_policy => :strict,
        :max_lag_boundary_relative_error => 0.05,
        :lag_min_gyroperiods => 0.015625,
        :lag_max_gyroperiods => 5.0,
        :lag_stride_gyroperiods => nothing,
        :min_lag_steps => 1,
        :max_lag_steps => nothing,
        :lag_step_stride => 1,
        :n_lag_samples => 40,
    )
    @test_throws ErrorException resolve_lag_grid(cfg, t_gp)

    nearest_cfg = copy(cfg)
    nearest_cfg[:lag_boundary_policy] = :nearest
    lag_grid = resolve_lag_grid(nearest_cfg, t_gp)
    @test first(lag_grid.tau_gyroperiods) ≈ 0.016087419809019843
    @test first(lag_grid.requested_tau_gyroperiods) ≈ 0.016087419809019843
    @test lag_grid.effective_lag_min_gyroperiods ≈ 0.016087419809019843
    @test lag_grid.lag_boundary_policy == :nearest

    too_tight = copy(nearest_cfg)
    too_tight[:max_lag_boundary_relative_error] = 0.01
    @test_throws ErrorException resolve_lag_grid(too_tight, t_gp)

    far_below = copy(nearest_cfg)
    far_below[:lag_min_gyroperiods] = 0.001
    @test_throws ErrorException resolve_lag_grid(far_below, t_gp)

    max_nearest = copy(nearest_cfg)
    max_nearest[:lag_min_gyroperiods] = 0.016087419809019843
    max_nearest[:lag_max_gyroperiods] = last(t_gp) + 0.25 * 0.016087419809019843
    max_grid = resolve_lag_grid(max_nearest, t_gp)
    @test last(max_grid.tau_gyroperiods) ≈ last(t_gp)

    max_far = copy(max_nearest)
    max_far[:lag_max_gyroperiods] = last(t_gp) + 2.0 * 0.016087419809019843
    @test_throws ErrorException resolve_lag_grid(max_far, t_gp)
end

@testset "lag range policies" begin
    t_gp_a = collect(0.0:0.25:5.0)
    t_gp_b = collect(0.0:0.30:4.8)
    cfg = Dict{Symbol, Any}(
        :lag_mode => :uniform_samples,
        :lag_range_policy => :first_cache_step,
        :lag_boundary_policy => :strict,
        :max_lag_boundary_relative_error => 0.0,
        :lag_min_gyroperiods => nothing,
        :lag_max_gyroperiods => 1.0,
        :lag_stride_gyroperiods => nothing,
        :min_lag_steps => 1,
        :max_lag_steps => nothing,
        :lag_step_stride => 1,
        :n_lag_samples => 4,
    )
    first_grid = resolve_lag_grid(cfg, t_gp_b)
    @test first_grid.effective_lag_min_gyroperiods ≈ 0.30

    bad_first = copy(cfg)
    bad_first[:lag_min_gyroperiods] = 0.25
    @test_throws ErrorException resolve_lag_grid(bad_first, t_gp_b)

    common_cfg = copy(cfg)
    common_cfg[:lag_range_policy] = :common_cache_intersection
    common_cfg[:lag_min_gyroperiods] = 0.25
    common_cfg[:lag_max_gyroperiods] = 5.0
    common_cfg[:common_cache_lag_min_gyroperiods] = max(lag_cache_summary(t_gp_a).cache_min_gp, lag_cache_summary(t_gp_b).cache_min_gp)
    common_cfg[:common_cache_lag_max_gyroperiods] = min(lag_cache_summary(t_gp_a).cache_max_gp, lag_cache_summary(t_gp_b).cache_max_gp)
    common_cfg[:lag_comparison_group_identity] = "test-group"
    common_grid = resolve_lag_grid(common_cfg, t_gp_a)
    @test common_grid.effective_lag_min_gyroperiods ≈ 0.30
    @test common_grid.effective_lag_max_gyroperiods ≈ 4.8
    @test common_grid.lag_comparison_group_identity == "test-group"
    @test common_grid.common_requested_tau_gyroperiods[1] ≈ 0.30
    @test normalize_lag_common_scope("reference-group") == :reference_group
    @test normalize_lag_common_scope("campaign") == :campaign
    @test_throws ErrorException normalize_lag_common_scope("mode")

    no_overlap = copy(common_cfg)
    no_overlap[:common_cache_lag_min_gyroperiods] = 5.0
    no_overlap[:common_cache_lag_max_gyroperiods] = 4.0
    @test_throws ErrorException resolve_lag_grid(no_overlap, t_gp_a)

    collapsed = copy(common_cfg)
    collapsed[:n_lag_samples] = 20
    collapsed[:lag_min_gyroperiods] = 0.31
    collapsed[:lag_max_gyroperiods] = 0.37
    @test_throws ErrorException resolve_lag_grid(collapsed, t_gp_a)
end

@testset "time axes" begin
    t_s = [0.0, 0.5, 1.0]
    Omega0 = 4.0
    t_norm = t_s .* Omega0
    t_gp = t_gyroperiods_from_axes(t_s, t_norm)
    @test validate_time_axes(t_s, t_norm, t_gp) == 3
    @test validate_time_axes(t_s, t_norm, t_gp; require_uniform=true) == 3
    @test_throws ErrorException validate_time_axes([0.0, 0.5, 0.4], t_norm, t_gp)
    @test_throws ErrorException validate_time_axes([0.0, 0.5, 1.2], [0.0, 2.0, 4.8], [0.0, 2.0 / (2pi), 4.8 / (2pi)]; require_uniform=true)
end

@testset "shared reference metadata" begin
    Bx_total = fill(3.0, 2, 2, 2)
    By_total = fill(4.0, 2, 2, 2)
    Bz_total = fill(0.0, 2, 2, 2)
    @test reference_B0_T(Bx_total, By_total, Bz_total) ≈ 5.0
    ref = campaign_time_reference(5.0, 2.0; source_mode="total", source_path="total.h5", source_identity="total.h5", field_subset=(2, 2, 2))
    @test ref.B0_reference_T ≈ 5.0
    @test ref.Omega0_reference_s_inv ≈ 2.0
    @test ref.time_reference_source_mode == "total"
    @test ref.time_reference_source_path == "total.h5"
    @test ref.reference_gyroperiod_s ≈ pi
end

@testset "direct runner reference guard" begin
    ref = campaign_time_reference(5.0, 2.0; source_mode="total", source_path="total.h5")
    @test require_explicit_campaign_time_reference("total", ref; caller="test") === ref
    for mode in ("alfven", "fast", "slow", "unknown", "total")
        @test_throws ErrorException require_explicit_campaign_time_reference(mode, nothing; caller="test")
    end
end

@testset "structured time validation" begin
    t_gp = [0.0, 0.25, 0.5, 0.75, 0.85]
    t_norm = 2.0 .* pi .* t_gp
    t_s = t_norm ./ 4.0
    result = validate_time_axes_result(t_s, t_norm, t_gp; require_uniform=true)
    @test !result.valid
    @test result.reason == :nonuniform_time_axis

    bad_norm = copy(t_norm)
    bad_norm[end] += 0.1
    result = validate_time_axes_result(t_s, bad_norm, t_gp; require_uniform=true)
    @test !result.valid
    @test result.reason == :inconsistent_time_units

    result = validate_time_axes_result(t_s[1:end-1], t_norm, t_gp; require_uniform=true)
    @test !result.valid
    @test result.reason == :time_length_mismatch
end

@testset "gpu production defaults" begin
    config_text = read(joinpath(@__DIR__, "..", "configs", "run_config.example.toml"), String)
    @test occursin("precision = \"Float32\"", config_text)
    @test occursin("trajectory_output_precision = \"Float32\"", config_text)
    @test occursin("compute_backend = \"gpu\"", config_text)
    @test occursin("compute_precision = \"Float32\"", config_text)
    @test occursin("accumulator_precision = \"Float32\"", config_text)
    @test occursin("gpu_lag_batch_size = 4", config_text)
    @test occursin("gpu_memory_fraction = 0.75", config_text)
    @test occursin("async_cache_writer = true", config_text)
    @test occursin("cache_writer_buffer_count = 2", config_text)
    @test occursin("gpu_pipeline_buffers = 2", config_text)
    @test occursin("save_raw_energy_snapshots = false", config_text)
end
