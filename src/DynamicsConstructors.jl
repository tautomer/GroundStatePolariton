using StaticArrays

abstract type Particles end
abstract type Bath end
abstract type Cache end
abstract type Particles1D <: Particles end
abstract type ParticlesND <: Particles end
abstract type Bath1D <: Bath end

struct Parameters
    temperature::Float64
    Δt::Float64
    z::Float64
    τ::Float64
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
    cosθ::Vector{Float64}
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

mutable struct Langevin <: Bath
    γ::Float64
    σ::Float64
    halfΔt2γ::Float64
    ran::MVector{3, Float64}
    dtγ::Float64
    dtσ::Float64
    dt2by2m::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct LangevinCache <: Cache
    cacheMol1::Vector{Float64}
    cacheMol2::Vector{Float64}
end

mutable struct SystemBathCache <: Cache
    cacheMol1::Vector{Float64}
    cacheBath::Vector{Float64}
end