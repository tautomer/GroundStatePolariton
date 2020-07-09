using StaticArrays

abstract type Particles end
abstract type DynamicsParameters end
abstract type Bath end
abstract type Cache end
abstract type Particles1D <: Particles end
abstract type ParticlesND <: Particles end
abstract type Bath1D <: Bath end
abstract type Langevin <: Bath1D end

struct Parameters <: DynamicsParameters
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

struct ReducedModelParameters <: DynamicsParameters
    temperature::Float64
    Δt::Float64
    nMol::Int32
end
mutable struct ReducedModelParticle <: Particles1D
    label::Vector{String}
    σ::Float64
    dtby2m::Float64
    x::Vector{Float64}
    f::Vector{Float64}
    v::Vector{Float64}
    cosθ::Vector{Float64}
    sumCosθ::Float64
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

mutable struct LangevinFull <: Langevin
    γ::Float64
    σ::Float64
    halfΔt2γ::Float64
    dtγ::Float64
    dtσ::Float64
    dt2by2m::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct LangevinModes <: Langevin
    γ::Float64
    σ::Float64
    halfΔt2γ::Float64
    dtγ::Float64
    dtσ::Float64
    dt2by2m::Float64
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct LangevinCache <: Cache
    ran1::Vector{Float64}
    ran2::Vector{Float64}
    cacheMol1::Vector{Float64}
    cacheMol2::Vector{Float64}
end

mutable struct SystemBathCache <: Cache
    cacheMol1::Vector{Float64}
    cacheBath::Vector{Float64}
end