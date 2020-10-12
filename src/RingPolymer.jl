module RingPolymer
export RPMDArrays, ringPolymerSetup, normalModeTransformation!, ringPolymerEvolution!
using LinearAlgebra: eigvals, eigvecs, mul!
push!(LOAD_PATH, pwd())
if isdefined(@__MODULE__,:LanguageServer)
    include("DynamicsConstructors.jl")
    using ..DynamicsConstructors
else
    using DynamicsConstructors
end

struct RPMDArrays
    ωₖ::Vector{Float64}
    tnm::Array{Float64}
    tnmi::Array{Float64}
    freerp::Array{Float64}
end

function ringPolymerSetup(nb::Integer, dt::T, temp::T) where T<:Real
    # ωₙ = 1 / βₙ
    ωₙ = nb * temp
    if nb > 1
        ωₖ, tnm, tnmi = ringNormalMode(nb)
    else
        ωₖ = [1.0]
        tnm = [1.0]
        tnmi = [1.0]
    end
    ωₖ *= ωₙ
    freerp = getFreeRPMatrix(nb, dt, ωₖ)
    rpmd = RPMDArrays(ωₖ, tnm, tnmi, freerp)
    return rpmd
end

"""
    function ringNormalMode(nb::Integer)

Function to compute the normal mode frequencies and transformation matrices.
All values can also be computed analytically. 

    twoPiByN = 2pi / nb
    sqrtInvN = sqrt(1.0/nb)
    sqrtInv2oN = sqrt(2.0/nb)
    ωₖ = Vector{Float64}(undef, nb)
    tnmi = Matrix{Float64}(undef, nb, nb)
    if nb % 2 == 0
        halfN = nb ÷ 2
        for i in 1:nb
            tnmi[i, halfN+1] = sqrtInvN * (-1)^i
        end
    else
        halfN = nb ÷ 2 + 1
    end
    for i in 1:nb
        tmp = twoPiByN * i
        ωₖ[i] = 2 * sin((i-1)*pi/nb)
        tnmi[i, 1] = sqrtInvN
        for j in 2:halfN
            tnmi[i, j] = sqrtInv2oN * cos(tmp*(j-1))
        end
        for j in halfN+2:nb
            # println(i, " ", j, " ", sin(tmp*(j-1)))
            tnmi[i, j] = sqrtInv2oN * sin(tmp*(j-1))
        end
    end
    tnm = tnmi'
"""
function ringNormalMode(nb::Integer)
    # build the matrix
    m = zeros(nb, nb)
    m[1, 1] = 2.0
    m[1, 2] = -1.0
    m[1, nb] = -1.0
    for i in 2:nb-1
        m[i, i-1] = -1.0
        m[i, i] = 2.0
        m[i, i+1] = -1.0
    end
    m[nb, 1] = -1.0
    m[nb, nb-1] = -1.0
    m[nb, nb] = 2.0
    # get ωₖ's (note: ωₙ hasn't been included here)
    ωₖ = sqrt.(abs.(eigvals(m)))
    # get inverse normal mode transformation matrix
    tnmi = eigvecs(m)
    # `DEPEV` can give wrong phase
    # it doesn't seem to mess anything up, but I'm going to fix it anyway
    if tnmi[1, 1] < 0
        @. tnmi *= -1.0
    end
    # transpose to get the forward NM transformation matrix
    tnm = tnmi'
    return ωₖ, tnm, tnmi
end

"""
    function getFreeRPMatrix(nb::Integer, dt::T, ω::AbstractVector{T}
        ) where T<:Real

Compute the monodromy matrix for ring polymer position evolution.
Note the matrix here does not contain the masses.
"""
function getFreeRPMatrix(nb::Integer, dt::T, ω::AbstractVector{T}
    ) where T<:AbstractFloat

    a = Matrix{Float64}(undef, 4, nb)
    # centroid
    a[1, 1] = 1.0
    a[2, 1] = 0.0
    a[3, 1] = dt
    a[4, 1] = 1.0
    # other modes
    for i in 2:nb
        ωi = ω[i]
        ωt = ωi * dt
        sinωt = sin(ωt)
        cosωt = cos(ωt)
        a[1, i] = cosωt
        # multiply masses for this when using for evolution
        a[2, i] = -ωi * sinωt
        # divide masses for this when using for evolution
        a[3, i] = sinωt / ωi
        a[4, i] = cosωt
    end
    return a
end

"""
    function normalModeTransformation(x::T, p::T, xt::T, pt::T, tm::T) where T<:
        AbstractMatrix{T1} where T1<:Real

Perform forward normal mode transformation with x, p being the primitive
variables and xt, pt as the transformed ones and backward transformation with
x, p being the transformed arrays.
"""
function normalModeTransformation!(x::T, p::T, xt::T, pt::T, tm::T) where T<:
    AbstractMatrix{T1} where T1<:Real

    mul!(xt, tm, x)
    mul!(pt, tm, p)
    # xt .= tm * x
    # pt .= tm * p
end

function normalModeTransformation!(x::T, p::T, xt::T, pt::T, tm::T1) where {T<:
    AbstractMatrix{T2}, T1<:AbstractVector{T2}} where T2<:Real

    xt .= x
    pt .= p
end

function ringPolymerEvolution!(p::RPMDParticle, ::uncnstr) 

    @inbounds @simd for i in eachindex(p.m)
        @inbounds @simd for j in eachindex(1:p.nb)
            @views mono = p.freerp[:, j]
            p.xnm[j, i], p.vnm[j, i] = beadUpdate(p.xnm[j ,i], p.vnm[j, i],
                p.m[i], mono)
        end
    end
end

function ringPolymerEvolution!(p::RPMDParticle, ::cnstr) 

    ringPolymerEvolution!(p, Val(:uncnstr))
    p.vnm[1, 1] = 0.0
    p.xnm[1, 1] = 0.0
end

function beadUpdate(x::T, p::T, m::T, mono::AbstractVector{T}) where T<:Real

    newv = mono[1] * p + mono[2] * x * m
    x = mono[3] * p / m + mono[4] * x
    v = newv
    return x, v
end

end
# temp = 300.0 / 315775.0
# dt = 16.0
# mass = 1836.0
# nb = 16
# rpmd = ringPolymerSetup(nb, dt, temp)
# freq = 1500.0 / 219474.63068
# force(x) = -0.08576034929437469 * x
# 
# using Statistics: mean
# x = repeat([1.0], nb)
# v = repeat([5e-4], nb)
# v .*= mass
# f = force.(x)
# xt = copy(x)
# vt = copy(v)
# vout = open("v2", "w")
# xout = open("x2", "w")
# for i in 1:1250
#     @. v += 0.5 * f * dt
#     xt .= rpmd.tnm * x
#     vt .= rpmd.tnm * v
#     for j in 1:4
#         newvt = rpmd.freerp[1, j] * vt[j] + rpmd.freerp[2, j] * xt[j]
#         xt[j] = rpmd.freerp[3, j] * vt[j] + rpmd.freerp[4, j] * xt[j]
#         vt[j] = newvt
#     end
#     x .= rpmd.tnmi * xt
#     v .= rpmd.tnmi * vt
#     @. f = force(x)
#     @. v += 0.5 * f * dt
#     println(vout, i*dt, " ", mean(v), " ", v[1], " ", v[2], " ", v[3], " ", v[4])
#     println(xout, i*dt, " ", mean(x), " ", x[1], " ", x[2], " ", x[3], " ", x[4])
# end
# close(vout)
# close(xout)