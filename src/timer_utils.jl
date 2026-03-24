# src/timer_utils.jl
module TimerUtils

export @timeit, print_timer_summary, reset_timer!, export_timer_summary

const timings = Dict{String, Float64}()
const call_counts = Dict{String, Int}()
const order = Vector{String}()
const timer_lock = ReentrantLock()

macro timeit(name, expr)
    return quote
        local t0 = time_ns()
        local val = $(esc(expr))
        local t1 = time_ns()
        local dt = (t1 - t0) / 1e9
        lock(TimerUtils.timer_lock) do
            if !haskey(TimerUtils.timings, $name)
                TimerUtils.timings[$name] = 0.0
                TimerUtils.call_counts[$name] = 0
                push!(TimerUtils.order, $name)
            end
            TimerUtils.timings[$name] += dt
            TimerUtils.call_counts[$name] += 1
        end
        val
    end
end

function reset_timer!()
    lock(timer_lock) do
        empty!(timings)
        empty!(call_counts)
        empty!(order)
    end
end

function format_duration(t_sec::Float64)
    days = floor(Int, t_sec / 86400)
    rem = t_sec - days * 86400
    hours = floor(Int, rem / 3600)
    rem -= hours * 3600
    mins = floor(Int, rem / 60)
    secs = rem - mins * 60
    
    parts = String[]
    if days > 0
        push!(parts, "$(days) days")
    end
    if days > 0 || hours > 0
        push!(parts, "$(hours)h")
    end
    if days > 0 || hours > 0 || mins > 0
        push!(parts, "$(mins)min")
    end
    
    sec_str = "$(round(secs, digits=3))s"
    
    if isempty(parts)
        return sec_str
    else
        return join(parts, ", ") * " and " * sec_str
    end
end

function _generate_summary_string(title="ROUTINE PERFORMANCE & TIMING SUMMARY REPORT")
    total_time = isempty(timings) ? 0.0 : sum(values(timings))
    s = "\n" * "="^65 * "\n"
    s *= " ⏱️  " * title * "\n"
    s *= "="^65 * "\n"
    s *= rpad("Operation Name", 35) * rpad("Calls", 10) * rpad("Time (s)", 12) * "Relative\n"
    s *= "-"^65 * "\n"
    for name in order
        t = timings[name]
        c = call_counts[name]
        rel = total_time > 0 ? (t / total_time) * 100 : 0.0
        t_str = string(round(t, digits=3))
        rel_str = string(round(rel, digits=1), "%")
        s *= rpad(name, 35) * rpad(string(c), 10) * rpad(t_str, 12) * rel_str * "\n"
    end
    s *= "-"^65 * "\n"
    s *= rpad("Total Tracked Physics Execution", 40) * format_duration(total_time) * "\n"
    s *= "="^65 * "\n"
    return s
end

function print_timer_summary(title="ROUTINE PERFORMANCE & TIMING SUMMARY REPORT")
    print(_generate_summary_string(title))
end

function export_timer_summary(filepath::String, title="ROUTINE PERFORMANCE & TIMING SUMMARY REPORT")
    open(filepath, "a") do f
        write(f, _generate_summary_string(title))
    end
end

end # module
