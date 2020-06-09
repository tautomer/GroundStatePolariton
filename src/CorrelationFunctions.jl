push!(LOAD_PATH, "./")
module CorrelationFunctions
using LinearAlgebra: dot
using Statistics: mean

"""
    projection(x1::Float64, x2::Float64, axis::Vector{Float64}})

Project primitive (R,q) or (v_R,v_q) onto the first normal mode
"""
function projection(x::Array, axis::Vector{Float64})
    if isa(x[1], Array)
        tmp = [mean(i) for i in x]
    else
        tmp = x
    end
    if length(tmp) == 1
        return tmp[1]
    else
        return dot(tmp, axis)
    end
end

function heaviSide(x)
    if x >= 0
        h = 1
    else
        h = 0
    end
    return h
end

function fluxSide(corrFS, v0, q::Float64)
    side = heaviSide(q)
    corrFS += v0 * side
    return corrFS
end

function fluxSide(corrFS, v0, q::Array)
    side = heaviSide.(q)
    corrFS += v0 * side
    return corrFS
end

end
