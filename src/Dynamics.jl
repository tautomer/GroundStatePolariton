module Dynamics
using Random
using StaticArrays

export Parameters, ClassicalParticle, ClassicalBathMode, QuantumBathMode, QuantumParticle

abstract type Particles end
abstract type Particles1D <: Particles end
abstract type ParticlesND <: Particles end
abstract type Bath1D <: Particles1D end

struct Parameters
    temperature::Float64
    Δt::Float64
    nParticle::Int32
    nMol::Int32
    nBath::Int32
    beadMol::Int16
    beadPho::Int16
    beadBath::Int16
end

mutable struct ClassicalParticle <: Particles1D
    n::Int32
    label::Vector{String}
    m::Vector{Float64}
    σ::Vector{Float64}
    x::Vector{Float64}
    f::Vector{Float64}
    dtby2m::Vector{Float64}
    v::Vector{Float64}
end

mutable struct ClassicalBathMode <: Bath1D
    n::Int16
    m::Float64
    σ::Float64
    ω::Vector{Float64}
    c::Vector{Float64}
    mω2::Vector{Float64}
    c_mω2::Vector{Float64}
    x::Vector{Float64}
    f::Vector{Float64}
    dtby2m::Float64
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
    c::Vector{Float64}
    ω::Vector{Float64}
    mω2::Vector{Float64}
    c_mω2::Vector{Float64}
    x::Array{Float64, 2}
    v::Array{Float64, 2}
end

function velocitySampling!(p::ClassicalBathMode, rng::AbstractRNG)
    Random.randn!(rng, p.v)
    p.v .*= p.σ
end

function velocitySampling!(p::ClassicalParticle, rng::AbstractRNG)
    Random.randn!(rng, p.v)
    p.v .*= p.σ
end

function velocitySampling(rng::AbstractRNG, param::Parameters, p::ParticlesND)
    n = param.beadMol
    v = Random.randn(rng, n) * sqrt(n * param.temperature / p.m)
    return v
end

function velocityUpdate!(p::Particles1D, b::Bath1D)
    @. p.v += p.f * p.dtby2m
    @. b.v += b.f * b.dtby2m
end

function velocityVelert!(p::Particles1D, b::Bath1D, param::Parameters,
    ∇u!::Function, cache::AbstractMatrix{T}; cnstr=true) where T <: AbstractFloat

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    p.x[1] = 0.0
    @. b.x += b.v * param.Δt
    force!(p, b, ∇u!, cache)
    velocityUpdate!(p, b)
end

function velocityVelert!(p::Particles1D, b::Bath1D, param::Parameters,
    ∇u!::Function, cache::AbstractMatrix{T}) where T <: AbstractFloat

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    force!(p, b, ∇u!, cache)
    velocityUpdate!(p, b)
end

function velocityVelert!(p::Particles1D, b::Bath1D, param::Parameters,
    ∇u!::Function, cache::AbstractMatrix{T}, ks::T, x0::T) where T <: AbstractFloat

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    force!(p, b, ∇u!, cache, ks, x0)
    velocityUpdate!(p, b)
end

function force!(p::Particles1D, b::Bath1D, ∇u!::Function, cache::AbstractMatrix{T}) where T<:AbstractFloat

    ∇u!(p.f, p.x, cache)
    index = 0
    for j in 1:p.n
        @inbounds @simd for i in 1:b.n
            index += 1
            tmp = b.c_mω2[i] * p.x[j] - b.x[index]
            b.f[index] = b.mω2[i] * tmp
            p.f[j] -= b.c[i] * tmp
        end
    end
end

function force!(p::Particles1D, b::Bath1D, ∇u!::Function, cache::AbstractMatrix{T},
    ks::T, x0::T) where T <: AbstractFloat

    ∇u!(p.f, p.x, cache)
    p.f[1] -= ks * (p.x[1]-x0)
    index = 0
    for j in 1:p.n
        @simd for i in 1:b.n
            index += 1
            tmp = b.c_mω2[i] * p.x[j] - b.x[index]
            b.f[index] = b.mω2[i] * tmp
            p.f[j] -= b.c[i] * tmp
        end
    end
end

end # module