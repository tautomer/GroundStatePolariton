module DynamicsConstructors
using StaticArrays, ExportAll

abstract type Particles end
abstract type DynamicsParameters end
abstract type Bath end
abstract type Langevin end
abstract type Cache end
abstract type Particles1D<:Particles end
abstract type ParticlesND<:Particles end
abstract type Bath1D<:Particles1D end

const cnstr = Val{:cnstr}
const uncnstr = Val{:uncnstr}
const restr = Val{:restr}
const Control = Union{cnstr, uncnstr, restr}

struct Parameters<:DynamicsParameters
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

struct ReducedModelParameters<:DynamicsParameters
    temperature::Float64
    Δt::Float64
    nMol::Int32
end
mutable struct ReducedModelParticle<:Particles1D
    label::Vector{String}
    σ::Float64
    dtby2m::Float64
    x::Vector{Float64}
    f::Vector{Float64}
    v::Vector{Float64}
    cosθ::Vector{Float64}
    sumCosθ::Float64
end

mutable struct FullSystemParticle<:Particles1D
    n::Int32
    label::Vector{String}
    m::Vector{Float64}
    σ::Vector{Float64}
    x::Vector{Float64}
    ks::Float64
    xi::Float64
    f::Vector{Float64}
    dtby2m::Vector{Float64}
    v::Vector{Float64}
    cosθ::Vector{Float64}
end

mutable struct RPMDParticle<:ParticlesND
    n::Int32
    nb::Int16
    label::Vector{String}
    m::Vector{Float64}
    σ::Vector{Float64}
    x::Matrix{Float64}
    ks::Float64
    xi::Float64
    xnm::Matrix{Float64}
    f::Matrix{Float64}
    dtby2m::Float64
    v::Matrix{Float64}
    vnm::Matrix{Float64}
    cosθ::Vector{Float64}
    tnm::VecOrMat{Float64}
    tnmi::VecOrMat{Float64}
    freerp::VecOrMat{Float64}
end

mutable struct ClassicalBathMode<:Bath1D
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

mutable struct LangevinFull<:Langevin
    γ::Float64
    σ::Float64
    halfΔt2γ::Float64
    dtγ::Float64
    dtσ::Float64
    dt2by2m::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct LangevinModes<:Langevin
    γ::Float64
    σ::Float64
    halfΔt2γ::Float64
    dtγ::Float64
    dtσ::Float64
    dt2by2m::Float64
    x::Vector{Float64}
    v::Vector{Float64}
end

mutable struct LangevinCache<:Cache
    ran1::Vector{Float64}
    ran2::Vector{Float64}
    cacheMol1::Vector{Float64}
    cacheMol2::Vector{Float64}
end

mutable struct SystemBathCache<:Cache
    cacheMol1::Vector{Float64}
    cacheBath::Vector{Float64}
end

@exportAll
end