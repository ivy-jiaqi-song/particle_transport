function gpu_lag_batches(lag_steps::Vector{Int}, requested_batch_size)
    batch_size = max(1, Int(requested_batch_size))
    return [first_index:min(first_index + batch_size - 1, length(lag_steps)) for first_index in 1:batch_size:length(lag_steps)]
end

function gpu_available_memory_bytes()
    try
        free_bytes, total_bytes = CUDA.available_memory(), CUDA.total_memory()
        return Int64(free_bytes), Int64(total_bytes)
    catch
        return typemax(Int64), typemax(Int64)
    end
end

function resolve_gpu_work_plan(cfg; particle_chunk_size::Integer, lag_count::Integer, n_bins::Integer=1, bytes_per_particle_lag::Integer=0, base_bytes::Integer=0)
    requested_lag_batch = max(1, Int(get(cfg, :gpu_lag_batch_size, lag_count)))
    memory_fraction = Float64(get(cfg, :gpu_memory_fraction, 0.75))
    0.0 < memory_fraction <= 1.0 || error("gpu_memory_fraction must be in (0, 1].")
    free_bytes, total_bytes = gpu_available_memory_bytes()
    safety_margin = min(Int64(256 * 1024^2), max(0, free_bytes ÷ 10))
    usable_bytes = min(round(Int64, memory_fraction * free_bytes), max(0, free_bytes - safety_margin))

    lag_batch = min(requested_lag_batch, lag_count)
    chunk_size = Int(particle_chunk_size)
    estimated = base_bytes + chunk_size * lag_batch * max(1, n_bins) * bytes_per_particle_lag
    while lag_batch > 1 && estimated > usable_bytes
        lag_batch = max(1, lag_batch ÷ 2)
        estimated = base_bytes + chunk_size * lag_batch * max(1, n_bins) * bytes_per_particle_lag
    end
    while chunk_size > 1 && estimated > usable_bytes
        chunk_size = max(1, chunk_size ÷ 2)
        estimated = base_bytes + chunk_size * lag_batch * max(1, n_bins) * bytes_per_particle_lag
    end
    estimated <= usable_bytes || error("GPU memory plan cannot fit minimum workload. estimated=$(estimated) usable=$(usable_bytes)")
    return (
        particle_chunk_size = chunk_size,
        lag_batch_size = lag_batch,
        estimated_peak_gpu_memory = estimated,
        available_gpu_memory = free_bytes,
        total_gpu_memory = total_bytes,
        memory_fraction = memory_fraction,
    )
end

function print_gpu_work_plan(label::AbstractString, plan)
    println(label, " GPU memory plan:")
    println("  particle_chunk_size      = ", plan.particle_chunk_size)
    println("  lag_batch_size           = ", plan.lag_batch_size)
    println("  estimated_peak_gpu_bytes = ", plan.estimated_peak_gpu_memory)
    println("  available_gpu_bytes      = ", plan.available_gpu_memory)
    println("  memory_fraction          = ", plan.memory_fraction)
    return nothing
end
