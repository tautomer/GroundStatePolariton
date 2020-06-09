push!(LOAD_PATH, "./")
module Dynamics
using Calculus
using Dierckx
using Random
using StaticArrays

export Parameters, ClassicalParticle, ClassicalBathMode, QuantumBathMode, QuantumParticle

abstract type Particles end
abstract type Particles1D <: Particles end
abstract type ParticlesND <: Particles end
abstract type Bath1D <: Particles end

struct Parameters
    temperature::Float64
    Δt::Float64
    nMol::Int16
    nPho::Int16
    nBath::Int16
    beadMol::Int16
    beadPho::Int16
    beadBath::Int16
end

mutable struct ClassicalParticle <: Particles1D
    label::Vector{String}
    m::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct ClassicalBathMode <: Bath1D
    n::Int16
    m::Float64
    ω::Vector{Float64}
    c::Vector{Float64}
    mω2::Vector{Float64}
    c_mω2::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct QuantumParticle <: ParticlesND
    label::Vector{String}
    m::Vector{Float64}
    x::Array{Float64, 2}
    v::Array{Float64, 2}
end

mutable struct QuantumBathMode <: ParticlesND
    m::Float64
    ω::Vector{Float64}
    c::Vector{Float64}
    mω2::Vector{Float64}
    c_mω2::Vector{Float64}
    x::Array{Float64, 2}
    v::Array{Float64, 2}
end

function velocitySampling(param::Parameters, p::Particles1D)
    v = Random.randn() * sqrt(param.temperature / p.m)
    return v
end

function velocitySampling(param::Parameters, p::ParticlesND)
    n = param.beadMol
    v = Random.randn(n) * sqrt(n * param.temperature / p.m)
    return v
end

function velocitySampling(param::Parameters, p::Bath1D)
    n = param.nBath
    v = Random.randn(n) * sqrt(n * param.temperature / p.m)
    return v
end

function velocityUpdate(param::Parameters, fc::Tuple, p::Particles1D, b::Bath1D=true)
    dt = param.Δt
    accdt1 = 0.5 * fc[1] / p.m * dt
    accdt2 = 0.5 * fc[2] / b.m * dt
    p.v += accdt1
    b.v += accdt2
    return p, b
end


function positionUpdate(param::Parameters, fc::Tuple, p::Particles1D, b::Bath1D; cnstr=true)
    dt = param.Δt
    p.x = 0.0
    b.x += b.v * dt + 0.5 * fc[2] / b.m * dt^2
    return p, b
end

function positionUpdate(param::Parameters, fc::Tuple, p::Particles1D, b::Bath1D)
    dt = param.Δt
    dt2 = 0.5 * dt^2
    p.x += p.v * dt + fc[1] / p.m * dt2
    b.x += b.v * dt + fc[2] / b.m * dt2
    return p, b
end

function force(u, q, b::Bath1D)
    fp = -Dierckx.derivative(u, q)
    fb = Vector{Float64}(undef, b.n)
    for i in 1:b.n
        tmp = q - b.c_mω2[i] * b.x[i]
        fb[i] = b.mω2[i] * tmp
        fp -= b.c[i] * tmp
    end
    return fp, fb
end
end
