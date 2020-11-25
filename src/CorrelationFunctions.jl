module CorrelationFunctions
# TODO RPMD version
using Statistics: mean

"""
    function getCentroid(a::AbstractVector{T}) where T<:Real

Return the first element (V₀ or q) for classical simulations.
"""
function getCentroid(a::AbstractVector{T}) where T<:Real
    return @views a[1]
end

"""
    function getCentroid(a::AbstractMatrix{T}) where T<:Real

Return the average of the first column (V₀ or q) for RPMD simulations.
"""
function getCentroid(a::AbstractMatrix{T}) where T<:Real
    @views c = mean(a[:, 1])
    return c
end

"""
    function heaviSide(x::Real)

Compute the side operator from input values
h(x) = 1 if x >= 0, or 0 if x < 0
"""
function heaviside(x::Real)
    if x >= 0.0
        h = 1.0
    else
        h = 0.0
    end
    return h
end

"""
    function fluxSide(corrFS::T, v0::T, q::T) where T<:AbstractFloat

Add the current flux-side value to the summation `corrFS`.
This is the float version, which is used to compute fs(t=0).
"""
function fluxSide(corrFS::T, v0::T, q::T) where T<:AbstractFloat
    corrFS += v0 * heaviside(q)
    return corrFS
end

"""
    function fluxSide!(corrFS::AbstractArray{T}, v0::T, q::AbstractArray{T}) where T<:AbstractFloat

Add the current flux-side value to the summation `corrFS`.
This is the vector version, which is used to compute fs(t).
Replacing the vectorized form with a simple loop is 1 ns faster, but is it worth?
"""
function fluxSide!(corrFS::AbstractArray{T}, v0::T, q::AbstractArray{T}
    ) where T<:AbstractFloat
    @. corrFS += v0 * heaviside(q)
end

function normalize!(fs::AbstractVector{T}, fs0::T) where T<:Real
    @inbounds @simd for i in eachindex(fs)
        fs[i] /= fs0
    end
end

end # module
