mutable struct AsyncPhaseSpaceSlot{Tout}
    id::Int
    dpacked::Any
    host::Matrix{Tout}
    save_idx::Int
    alive_fraction::Float64
end

function maybe_pin_host_buffer(buffer)
    try
        return CUDA.pin(buffer)
    catch
        return buffer
    end
end

function create_async_phase_space_writer(path::AbstractString, n_particles::Integer, nsave::Integer, ::Type{Tout}, buffer_count::Integer) where {Tout<:AbstractFloat}
    buffer_count >= 2 || error("cache_writer_buffer_count must be >= 2 when async_cache_writer is enabled.")
    file = h5open(path, "w")
    positions = create_dataset(file, "positions", Tout, (n_particles, 3, nsave))
    momenta = create_dataset(file, "momenta", Tout, (n_particles, 3, nsave))
    free_slots = Channel{AsyncPhaseSpaceSlot{Tout}}(buffer_count)
    write_queue = Channel{AsyncPhaseSpaceSlot{Tout}}(buffer_count)
    for slot_id in 1:buffer_count
        put!(free_slots, AsyncPhaseSpaceSlot{Tout}(slot_id, CUDA.fill(Tout(NaN), n_particles, 6), maybe_pin_host_buffer(Array{Tout}(undef, n_particles, 6)), 0, NaN))
    end
    writer_task = @async begin
        pending = Dict{Int, AsyncPhaseSpaceSlot{Tout}}()
        next_write = 1
        while next_write <= nsave
            slot = take!(write_queue)
            pending[slot.save_idx] = slot
            while haskey(pending, next_write)
                ready = pop!(pending, next_write)
                positions[:, 1, next_write] = ready.host[:, 1]
                positions[:, 2, next_write] = ready.host[:, 2]
                positions[:, 3, next_write] = ready.host[:, 3]
                momenta[:, 1, next_write] = ready.host[:, 4]
                momenta[:, 2, next_write] = ready.host[:, 5]
                momenta[:, 3, next_write] = ready.host[:, 6]
                put!(free_slots, ready)
                next_write += 1
            end
        end
    end
    return (file=file, positions=positions, momenta=momenta, outtype=Tout, free_slots=free_slots, write_queue=write_queue, writer_task=writer_task, buffer_count=buffer_count, async=true)
end

function enqueue_phase_space_step!(writer, save_idx::Integer, alive_fraction::Float64, pack_callback)
    slot = take!(writer.free_slots)
    slot.save_idx = Int(save_idx)
    slot.alive_fraction = alive_fraction
    pack_callback(slot.dpacked)
    copyto!(slot.host, Array(slot.dpacked))
    put!(writer.write_queue, slot)
    return nothing
end

function finish_async_phase_space_writer!(writer)
    wait(writer.writer_task)
    return nothing
end

mutable struct AsyncVectorSlot{Tout}
    id::Int
    device::Any
    host::Vector{Tout}
    save_idx::Int
end

function create_async_vector_writer(path::AbstractString, dataset_name::AbstractString, n_particles::Integer, nsave::Integer, ::Type{Tout}, buffer_count::Integer) where {Tout<:AbstractFloat}
    buffer_count >= 2 || error("cache_writer_buffer_count must be >= 2 when async_cache_writer is enabled.")
    file = h5open(path, "w")
    dataset = create_dataset(file, dataset_name, Tout, (n_particles, nsave))
    free_slots = Channel{AsyncVectorSlot{Tout}}(buffer_count)
    write_queue = Channel{AsyncVectorSlot{Tout}}(buffer_count)
    for slot_id in 1:buffer_count
        put!(free_slots, AsyncVectorSlot{Tout}(slot_id, CUDA.fill(Tout(NaN), n_particles), maybe_pin_host_buffer(Vector{Tout}(undef, n_particles)), 0))
    end
    writer_task = @async begin
        pending = Dict{Int, AsyncVectorSlot{Tout}}()
        next_write = 1
        while next_write <= nsave
            slot = take!(write_queue)
            pending[slot.save_idx] = slot
            while haskey(pending, next_write)
                ready = pop!(pending, next_write)
                dataset[:, next_write] = ready.host
                put!(free_slots, ready)
                next_write += 1
            end
        end
    end
    return (file=file, dataset=dataset, outtype=Tout, free_slots=free_slots, write_queue=write_queue, writer_task=writer_task, buffer_count=buffer_count, async=true)
end

function enqueue_vector_step!(writer, save_idx::Integer, fill_callback)
    slot = take!(writer.free_slots)
    slot.save_idx = Int(save_idx)
    fill_callback(slot.device)
    copyto!(slot.host, Array(slot.device))
    put!(writer.write_queue, slot)
    return nothing
end

function finish_async_vector_writer!(writer)
    wait(writer.writer_task)
    return nothing
end
