module Dynamics
using Random

include("DynamicsConstructors.jl")
export Parameters, ClassicalParticle, ClassicalBathMode, QuantumParticle

"""
    function andersen!(p::ClassicalParticle, b::ClassicalBathMode, cf::T,
        cache::DynamicsCache) where T<:Real

Perform Andersen thermostat for every 1/ν steps. `cf` is the collision frequency.
Removing COM velocity is currently missing.
"""
function andersen!(p::ClassicalParticle, b::ClassicalBathMode, cf::T,
    rng::AbstractRNG, cache::SystemBathCache) where T<:Real
    Random.rand!(rng, cache.cacheMol1)
    @inbounds @simd for i in eachindex(p.v)
        if cache.cacheMol1[i] < cf
            p.v[i] = Random.randn(rng) * p.σ[i]
        end
    end
    Random.rand!(rng, cache.cacheBath)
    @inbounds @simd for i in eachindex(b.v)
        if cache.cacheBath[i] < cf
            b.v[i] = Random.randn(rng) * b.σ
        end
    end
end

function velocitySampling!(p::ClassicalParticle, b::ClassicalBathMode, rng::AbstractRNG)
    Random.randn!(rng, p.v)
    p.v .*= p.σ
    Random.randn!(rng, b.v)
    b.v .*= b.σ
end

function velocitySampling!(p::ClassicalParticle, b::Langevin, rng::AbstractRNG)
    Random.randn!(rng, p.v)
    p.v .*= p.σ
end

function velocitySampling(rng::AbstractRNG, param::Parameters, p::ParticlesND)
    n = param.beadMol
    v = Random.randn(rng, n) * sqrt(n * param.temperature / p.m)
    return v
end

function velocityUpdate!(p::ClassicalParticle, b::Bath1D)
    @. p.v += p.f * p.dtby2m
    @. b.v += b.f * b.dtby2m
end

const invSqrt3 = 1.0 / sqrt(3.0)
# γ = 0.0091126705341906
# σ = sqrt(2γ/1052.58/1836.0)
"""
    function randomForce(prefac::T, r1::T, r2::T) where T<:Float64

Compute the random force in Langevin modified Velocity Verlet. 
f = σ * Δt^1.5 * (0.5 * r1 + r2 * 0.5 / √3), where `r1` and `r2` are gaussian
random numbers. The prefactor `prefac` is defined as 0.5 * σ * Δt.
"""
function randomForce(prefac::T, r1::T, r2::T) where T<:Float64
    return prefac * (r1 + invSqrt3 * r2)
end

"""
    function positionUpdate(x::T, v::T, f::T, dt::T, lgv::Langevin) where T<:Real

Compute the new corrdinates for all molecules in Langevin modified Velocity Verlet. 
"""
function positionUpdate(x::T, v::T, f::T, r1::T, r2::T, dt::T, lgv::Langevin
    ) where T<:Real
    acc = lgv.dt2by2m[1] * f - lgv.halfΔt2γ * v + randomForce(lgv.dtσ,
        r1, r2)
    x += v * dt + acc
    return acc, x
end

"""

Compute the new corrdinates for all molecules in Langevin modified Velocity Verlet. 
"""
function velocityUpdate(v::T, f::T, r3::T, fOld::T, dtby2m::T, acc::T,
    lgv::Langevin) where T<:Real
    accLgv = lgv.σ * r3 - lgv.γ * acc - lgv.dtγ * v 
    v += (fOld+f) * dtby2m + accLgv
    return v
end

function velocityVerlet!(p::ClassicalParticle, lgv::Langevin, param::Parameters,
    rng::AbstractRNG, ∇u!::Function, cache::T, ::Val{:cnstr}) where T<:Cache

    # obatin all three random numbers at once
    Random.randn!(rng, cache.ran2)
    # copy current forces to cache
    copy!(cache.cacheMol1, p.f)
    # apply the constraint
    p.x[1] = 0.0
    p.v[1] = 0.0
    # update the position and velocities for each molecule
    index = 0
    @inbounds @simd for i in 2:p.n
        index += 2
        cache.cacheMol2[i], p.x[i] = positionUpdate(p.x[i], p.v[i], p.f[i],
            cache.ran2[index-1], cache.ran2[index], param.Δt, lgv)
    end
    # update the position for the photon
    p.x[end] += p.v[end] * param.Δt + p.f[end] * lgv.dt2by2m[end]
    # compute new forces
    ∇u!(p.f, p.x, p.cosθ)
    # update the velocity for each molecule
    @inbounds @simd for i in 2:p.n
        index += 1
        p.v[i] = velocityUpdate(p.v[i], p.f[i], cache.ran2[index],
            cache.cacheMol1[i], p.dtby2m[i], cache.cacheMol2[i], lgv)
    end
    # update the velocity for the photon
    p.v[end] += (cache.cacheMol1[end]+p.f[end]) * p.dtby2m[end]
end

function velocityVerlet!(p::ClassicalParticle, lgv::Langevin, param::Parameters,
    rng::AbstractRNG, ∇u!::Function, cache::T) where T<:Cache

    # obatin all three random numbers at once
    Random.randn!(rng, cache.ran1)
    # copy current forces to cache
    copy!(cache.cacheMol1, p.f)
    # update the position for the first molecule
    cache.cacheMol2[1], p.x[1] = positionUpdate(p.x[1], p.v[1], p.f[1],
        cache.ran1[1], cache.ran1[2], param.Δt, lgv)
    # update the position for the rest DOF
    @inbounds @simd for i in 2:param.nParticle
        p.x[i] += p.v[i] * param.Δt + p.f[i] * lgv.dt2by2m[i]
    end
    # compute new forces
    ∇u!(p.f, p.x, p.cosθ)
    # update the velocity for each molecule
    p.v[1] = velocityUpdate(p.v[1], p.f[1], cache.ran1[3], cache.cacheMol1[1],
        p.dtby2m[1], cache.cacheMol2[1], lgv)
    # update the velocity for the photon
    @inbounds @simd for i in 2:param.nParticle
        p.v[i] += (cache.cacheMol1[i]+p.f[i]) * p.dtby2m[i]
    end
end

function velocityVerlet!(p::Particles1D, b::Bath1D, param::Parameters,
    ∇u!::Function; cnstr=true)

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    p.x[1] = 0.0
    @. b.x += b.v * param.Δt
    force!(p, b, ∇u!)
    velocityUpdate!(p, b)
end

function velocityVerlet!(p::Particles1D, b::Bath1D, param::Parameters,
    ∇u!::Function)

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    force!(p, b, ∇u!)
    velocityUpdate!(p, b)
end

function velocityVerlet!(p::Particles1D, b::Bath1D, param::Parameters,
    ∇u!::Function, ks::T, x0::T) where T <: AbstractFloat

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    force!(p, b, ∇u!, ks, x0)
    velocityUpdate!(p, b)
end

function force!(p::Particles1D, b::Bath1D, ∇u!::Function)

    ∇u!(p.f, p.x)
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

function force!(p::Particles1D, b::Bath1D, ∇u!::Function, ks::T, x0::T) where T <: AbstractFloat

    ∇u!(p.f, p.x)
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

function force!(p::Particles1D, b::Langevin, ∇u!::Function)

    ∇u!(p.f, p.x)
end
"""
    function equilibration!(p::Particles1D, b::Bath1D, nst::Int64, rng::AbstractRNG,
        param::Parameters, ∇u!::Function, cache::SystemBathCache)

Equilibrate the particles in a system-bath model. We use an Andersen thermostat
to maintain a NVE ensemble. The first molecule is constrained on the top of the
barrier.
"""
function equilibration!(p::Particles1D, b::Bath1D, nst::Int64, rng::AbstractRNG,
    param::Parameters, ∇u!::Function, cache::SystemBathCache)

    @inbounds for j in 1:nst
        velocityVerlet!(p, b, param, ∇u!, cnstr=true)
        if j % param.τ == 0
            andersen!(p, b, param.z, rng, cache)
        end
    end
end

"""
    function equilibration!(p::Particles1D, b::Langevin, nst::Int64, rng::AbstractRNG,
        param::Parameters, ∇u!::Function, cache::SystemBathCache)

Equilibrate the particles with Langevin dynamics. The first molecule is
constrained on the top of the barrier.
"""
function equilibration!(p::Particles1D, b::Langevin, nst::Int64, rng::AbstractRNG,
    param::Parameters, ∇u!::Function, cache::LangevinCache)

    @inbounds for j in 1:nst
        velocityVerlet!(p, b, param, rng, ∇u!, cache, Val(:cnstr))
    end
end

end # module
