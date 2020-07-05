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

function velocityUpdate!(p::ClassicalParticle, b::Bath1D)
    @. p.v += p.f * p.dtby2m
    @. b.v += b.f * b.dtby2m
end

const invSqrt12 = 0.5 / sqrt(3.0)
# γ = 0.0091126705341906
# σ = sqrt(2γ/1052.58/1836.0)
function randomForce(prefac::T, r1::T, r2::T) where T<:Float64
    return prefac * (0.5 * r1 + invSqrt12 * r2)
end
function velocityVerlet!(p::ClassicalParticle, lgv::Lagevin, param::Parameters,
    rng::AbstractRNG, ∇u!::Function, cache::T, ::Val{:cnstr}) where T<:Cache

    Random.randn!(rng, lgv.ran)
    copy!(cache.cacheMol1, p.f)
    for i in eachindex(1:p.n)
        cache.cacheMol2[i] = lgv.dt2by2m[i] * p.f[i] - lgv.halfΔt2γ * p.v[i] +
            randomForce(lgv.dtσ, lgv.ran[1], lgv.ran[2])
        p.x[i] += p.v[i] * param.Δt + acc[i]
    end
    p.x[1] = 0.0
    p.x[end] += p.v[end] * param.Δt + p.f[end] * 8.0
    force!(p, b, ∇u!, cache)
    for i in eachindex(1:p.n)
        p.v[i] += (fold[i]+p.f[i]) * p.dtby2m[i] - 4.0 * γ * p.v[i] + 2.0 * σ * Random.randn() - γ * acc[i]
    end
    p.v[end] += (fold[end]+p.f[end]) * 2.0
end

function velocityVerlet!(p::ClassicalParticle, b::Lagevin, param::Parameters,
    rng::AbstractRNG, ∇u!::Function, cache::T) where T<:Cache
    γ = 0.0091126705341906
    σ = sqrt(2γ/1052.58/1836.0)
    r1 = 0.5Random.randn()
    r2 = 0.28867513459481288225457439025098Random.randn()
    copy!(fold, p.f)
    for i in eachindex(1:p.n)
        acc[i] = param.Δt^2 / 2.0 / 1836.0 * p.f[i] - 8.0 * γ * p.v[i] + 8.0 * σ * (r1+r2)
        p.x[i] += p.v[i] * param.Δt + acc[i]
    end
    p.x[end] += p.v[end] * param.Δt + p.f[end] * 8.0
    force!(p, b, ∇u!, cache)
    for i in eachindex(1:p.n)
        p.v[i] += (fold[i]+p.f[i]) * p.dtby2m[i] - 4.0 * γ * p.v[i] + 2.0 * σ * Random.randn() - γ * acc[i]
    end
    p.v[end] += (fold[end]+p.f[end]) * 2.0
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

function equilibration!(p::Particles1D, b::Bath1D, nst::Int64, rng::AbstractRNG,
    param::Parameters, ∇u!::Function, cache::SystemBathCache)

    @inbounds for j in 1:nst
        velocityVerlet!(p, b, param, ∇u!, cnstr=true)
        if j % param.τ == 0
            andersen!(p, b, param.z, rng, cache)
        end
    end
end

end # module