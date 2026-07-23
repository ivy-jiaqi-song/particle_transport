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
        :lag_min_gyroperiods => 0.2,
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
end

@testset "time axes" begin
    t_s = [0.0, 0.5, 1.0]
    Omega0 = 4.0
    t_norm = t_s .* Omega0
    t_gp = t_gyroperiods_from_axes(t_s, t_norm)
    @test validate_time_axes(t_s, t_norm, t_gp) == 3
    @test_throws ErrorException validate_time_axes([0.0, 0.5, 0.4], t_norm, t_gp)
end
