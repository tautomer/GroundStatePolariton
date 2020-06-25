module Auxiliary
"""
    Some physical constants useful to the simulation
"""
module Constants

export au2wn, au2ev, au2kelvin, au2kcal, amu2au, au2fs

au2wn = 219474.63068
au2ev = 27.2113961
au2kelvin = 3.15775e5
au2kcal = 627.509
amu2au = 1836.0
au2fs = 2.41888e-2

end

using .Constants
using Calculus
using Optim, LineSearches

"""
    normalModeAnalysis(x0::Array{Float64,1})

Return normal mode frequencies, eigenvectors and crossover temperature
"""
function normalModeAnalysis(u::Function, x0::Array{Float64,1}, omegaC::Float64)
    hf = Calculus.hessian(x -> u(x[1], x[2]))
    mat = hf(x0)
    # possibly a bug in Calculus package
    # when ω is fairly small, ∂²V/∂q² will be ZERO
    mat_og = copy(mat)
    mat[1, 1] /= amu2au
    mat[2, 1] /= sqrt(amu2au)
    mat[1, 2] = mat[2, 1]
    mat[2, 2] = omegaC^2
    lambda = Calculus.eigvals(mat)
    v = Calculus.eigvecs(mat)
    if lambda[1] < 0
        lambda[1] = -sqrt(-lambda[1])
        tC = -lambda[1] / (2pi) * au2kelvin
    else
        lambda[1] = sqrt(lambda[1])
        tC = NaN
    end
    lambda[2] = sqrt(lambda[2])
    return lambda*au2wn, v, tC
end

"""
    optPES(targetFunc::Function, x0::Array{Float64, 1}, algor; maxTries=5)

`optPES` will find a local minimum on the PES with Optim package
minimum location and value are returned
"""
function optPES(targetFunc::Function, x0::Array{Float64, 1}, algor;
    maxTries=5)

    n = 0
    local minLoc, minVal
    while n < maxTries
        result = optimize(x -> targetFunc(x[1], x[2]), x0, algor())
        minLoc = Optim.minimizer(result)
        minVal = Optim.minimum(result)
        if Optim.converged(result)
            if Optim.iterations(result) == 0
                println("Warning: Starting point is a stationary point.
                    Output $minLoc is not necessarily a minimum.")
            end
            break
        else
            x0 = minLoc
            n += 1
            if n > maxTries
                throw("Hit max number of tries in `optPES`.
                    Consider using different parameters.")
            end
        end
    end
    return minLoc, minVal
end

end

module Dynamics
using ForwardDiff: gradient!, GradientConfig
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
    nParticle::Int16
    nMol::Int16
    nBath::Int16
    beadMol::Int16
    beadPho::Int16
    beadBath::Int16
end

mutable struct ClassicalParticle <: Particles1D
    n::Int64
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
    pot::Function, cache::GradientConfig; cnstr=true)

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    p.x[1] = 0.0
    @. b.x += b.v * param.Δt
    force!(p, b, pot, cache)
    velocityUpdate!(p, b)
end

function velocityVelert!(p::Particles1D, b::Bath1D, param::Parameters,
    pot::Function, cache::GradientConfig)

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    force!(p, b, pot, cache)
    velocityUpdate!(p, b)
end

function velocityVelert!(p::Particles1D, b::Bath1D, param::Parameters,
    pot::Function, cache::GradientConfig, ks::T, x0::T) where T <: AbstractFloat

    velocityUpdate!(p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    force!(p, b, pot, cache, ks, x0)
    velocityUpdate!(p, b)
end

function force!(p::Particles1D, b::Bath1D, pot::Function, cache::GradientConfig)

    gradient!(p.f, pot, p.x, cache)
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

function force!(p::Particles1D, b::Bath1D, pot::Function, cache::GradientConfig,
    ks::T, x0::T) where T <: AbstractFloat

    gradient!(p.f, pot, p.x, cache)
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

end

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

function heaviSide(x::AbstractFloat)
    if x >= 0.0
        h = 1.0
    else
        h = 0.0
    end
    return h
end

function fluxSide(corrFS::T, v0::T, q::T) where T <: AbstractFloat
    corrFS += v0 * heaviSide(q)
    return corrFS
end

function fluxSide!(corrFS::AbstractArray{T}, v0::T, q::AbstractArray{T}
    ) where T <: AbstractFloat
    @. corrFS += v0 * heaviSide(q)
end

end

using DelimitedFiles
using Interpolations: LinearInterpolation, Line, AbstractExtrapolation
using Printf
using Random
using WHAM
using Statistics: mean, varm
using ForwardDiff: GradientConfig
using ..Dynamics
using ..Auxiliary
using .Auxiliary.Constants: au2wn, au2ev, au2kelvin, au2kcal, amu2au, au2fs
using ..CorrelationFunctions

const corr = CorrelationFunctions

"""
    function initialize(temp, freqCutoff, eta, ωc, chi)

Initialize most of the values, parameters and structs for the dynamics.
"""
# TODO better and more flexible way to handle input values
function initialize(temp::T, freqCutoff::T, eta::T, ωc::T, chi::T) where T<:Real
    # convert values to au, so we can keep more human-friendly values outside
    ωc /= au2ev
    temp /= au2kelvin
    nPhoton = 1
    nParticle = 50
    nMolecule = nParticle - nPhoton
    # number of bath modes per molecule! total number of bath mdoes = nMolecule * nBath
    nBath = 15
    dt = 4.0

    # compute coefficients for bath modes
    nb = real(nBath)
    # bath mode frequencies
    ω = -freqCutoff * log.((collect(1.0:nb).-0.5) / nb)
    # bath force constants
    mω2 = ω.^2 * amu2au
    # bath coupling strength
    c = sqrt(2eta * amu2au * freqCutoff / nb / pi) * ω
    # c/mω^2, for force evaluation
    c_mω2 = c ./ mω2

    # check the number of molecules and photons
    if nParticle != nMolecule # && chi != 0.0
        # χ != 0 
        if nParticle - nMolecule > 1
            println("Multi-modes not supported currently. Reduce to single-mode.")
            nMolecule = nParticle - 1
        end
        label = ["photon"]
        mass = [1.0]
    else
        if nMolecule > 1
            println("In no-coupling case, multi-molecule does not make sense. Reduce to single-molecule.")
            nParticle = 1
            nMolecule = 1
        end
        label = Vector{String}(undef, 0)
        mass = Vector{Float64}(undef, 0)
    end
    label = vcat(repeat(["mol"], nMolecule), label)
    mass = vcat(repeat([amu2au], nMolecule), mass)
    if nMolecule > 1
        nBathTotal = nMolecule * nBath
        dummy = Vector{Float64}(undef, nBathTotal)
    else
        nBathTotal = nBath
        dummy = ω
    end
    param = Dynamics.Parameters(temp, dt, nParticle, nMolecule, nBathTotal, 1, 1, 1)
    bath = Dynamics.ClassicalBathMode(nBath, amu2au, sqrt(temp/amu2au),
        ω, c, mω2, c_mω2, similar(dummy), similar(dummy), param.Δt/(2*amu2au),
        similar(dummy))
    mol = Dynamics.ClassicalParticle(nMolecule, label, mass, sqrt.(temp./mass),
        similar(mass), similar(mass), param.Δt./(2*mass), similar(mass))
    uTotal = constructPotential(pesMol, dipole, ωc, chi, nParticle, nMolecule)
    cache = GradientConfig(uTotal, similar(mass))
    return param, mol, bath, uTotal, cache
end

"""
    function getPES(pes="pes.txt"::String, dm="dm.txt"::String)

Read porential energy surface and dipole moment from text files and then get the
interpolated functions. The filenames are default to "pes.txt" and "dm.txt".
"""
function getPES(pes="pes.txt"::String, dm="dm.txt"::String)
    potentialRaw = readdlm(pes)
    dipoleRaw = readdlm(dm)
    function interpolate(x::AbstractVector{T}, y::AbstractVector{T}
        ) where T <: AbstractFloat
        # TODO beter way to get the range from an array assumed evenly spaced
        xrange = LinRange(x[1], x[end], length(x))
        return LinearInterpolation(xrange, y, extrapolation_bc=Line())
    end
    return interpolate(view(potentialRaw, :, 1), view(potentialRaw, :, 2)),
        interpolate(view(dipoleRaw, :, 1), view(dipoleRaw, :, 2))
end

"""
    function constructPotential(pesMol::T1, dipole::T1, omegaC::T2, chi::T2,
        nParticle::T2, nMolecule::T2) where {T1<:AbstractExtrapolation, T2<:Real}

Function to construct the total potential of the polaritonic system for different
number of Molecules and photons.

Note that the returned potential is only for calclating force purpose, so it is
inverted to avoid a "-" at each iteration.
"""
# TODO RPMD & multi-modes?
function constructPotential(pesMol::T1, dipole::T1, omegaC::T2, chi::T2,
    nParticle::T3, nMolecule::T3) where {T1<:AbstractExtrapolation,
    T2, T3<:Real}
    # compute the constants in advances. they can be expensive for many cycles
    kPho = 0.5 * omegaC^2
    # sqrt(2 / ω_c^3) * χ
    couple = sqrt(2/omegaC^3) * chi
    # construct an inverted total potential
    # TODO use 1 float number in this case for better performance
    @inline function uTotalOneD(x::AbstractVector{T}) where T<:Real
        return @fastmath -pesMol(x[1])
    end

    @inline function uTotalSingleMol(x::AbstractVector{T}) where T<:Real
        return @inbounds @fastmath -pesMol(x[1]) -
            kPho * (x[2] + couple*dipole(x[1]))^2
    end
    @inline function uTotalMultiMoltest(x::AbstractArray{T}) where T<:Real
        u = 0.0
        @inbounds @fastmath for i in 1:length(x)
            u -= pesMol(x[i]) 
        end
        return u 
    end
    # TODO I still think it's possible to get better performance with stuff like `sum` or vector operations
    @inline function uTotalMultiMol(x::AbstractArray{T}) where T<:Real
        # ∑μ = sum(dipole.(view(x, 1:20)))
        # u = sum(pesMol.(view(x, 1:20)))
        u = 0.0
        ∑μ = 0.0
        @inbounds @fastmath for i in 1:length(x)-1
            ∑μ += dipole(x[i])
            u -= pesMol(x[i]) 
        end
        return @fastmath u - kPho * (x[end] + couple*∑μ)^2
    end
    if nParticle == nMolecule
        # when there is no photon, we force the system to be 1D
        # we will still use an 1-element vector as the input
        # return uTotalMultiMoltest
        return uTotalOneD
    else
        if nMolecule == 1
            # single-molecule and single photon
            return uTotalSingleMol
        else
            # multi-molecules and single photon
            return uTotalMultiMol
        end
    end
end

function computeKappa(temp::T1, nTraj::T2, nStep::T2, ωc::T1, chi::T1
    ) where {T1<:Real, T2<:Integer}
    rng = Random.seed!(1233+Threads.threadid())

    param, mol, bath, uTotal, cache = initialize(temp, freqCutoff, eta, ωc, chi)

    fs0 = 0.0
    fs = zeros(nStep+1)
    q = zeros(nStep+1)
    # e = zeros(nStep+1)
    x0 = repeat([-1.7338], param.nParticle)
    x0[1] = 0.0
    x0[end] = -183.71710507478846(param.nMol-1)
    # x0 = zeros(length(mol.x))
    xb0 = Random.randn(param.nBath)
    for i in 1:nTraj
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        Dynamics.velocitySampling!(mol, rng)
        Dynamics.velocitySampling!(bath, rng)
        copy!(mol.x, x0)
        copy!(bath.x, xb0)
        Dynamics.force!(mol, bath, uTotal, cache)
        for j in 1:1000
            Dynamics.velocityVelert!(mol, bath, param, uTotal, cache,
                cnstr=true)
            if j % 50 == 0
                Dynamics.velocitySampling!(mol, rng)
                Dynamics.velocitySampling!(bath, rng)
            end
        end
        copy!(x0, mol.x)
        copy!(xb0, bath.x)
        v0 = mol.v[1] 
        q .= 0.0
        q[1] = v0
        # e[1] += reactiveEnergy(mol)
        # println(output, "# ", v0)
        fs0 = corr.fluxSide(fs0, v0, v0)
        for j in 1:nStep
            Dynamics.velocityVelert!(mol, bath, param, uTotal, cache)
            q[j+1] = mol.x[1]
            # e[j+1] += reactiveEnergy(mol)
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", mol.x[3], " ", mol.x[4])
            # println(output, j, " ", mol.x[1], " ", pesMol(mol.x[1]) + 0.5 * amu2au * mol.v[1]^2)
            # if q[j+1] * q[j] < 0
            #     println(output, "# here")
            # end
        end
        fs .+= q
        # close(output)
        # corr.fluxSide!(fs, v0, q)
    end

    # e /= nTraj + 0.0
    # open("check_energy.txt", "w") do io
    #     writedlm(io, e)
    # end

    return printKappa(fs, fs0, ωc, chi, temp, param)
end

function reactiveEnergy(p::Dynamics.ClassicalParticle)
    return pesMol(p.x[1]) + 918.0 * p.v[1]^2
end

function printKappa(fs::AbstractVector{T}, fs0::T, ωc::T, chi::T, temp::T,
    param::Dynamics.Parameters) where T <: AbstractFloat
    fs /= fs0
    flnm = string("fs_", ωc, "_", chi, "_", temp, "_", param.nMol, "_wbath.txt")
    fsOut = open(flnm, "w")
    @printf(fsOut, "# Thread ID %3d\n", Threads.threadid())
    @printf(fsOut, "# ω_c=%7.3e,χ=%6.4g \n", ωc, chi)
    for i in 1:length(fs)
        @printf(fsOut, "%5.2f %11.8g\n", (i-1)*param.Δt*2.4189e-2, fs[i])
    end
    close(fsOut)
    println("Results written to file: $flnm")
    return fs
end

function umbrellaSetup(temp::T, nw::Integer, ks::T, bound::Vector{T}) where T <: AbstractFloat
    sort!(bound)
    Δwindow = (bound[2]-bound[1]) / nw
    σ = sqrt(temp/ks)
    if σ < Δwindow
        println("""Warning: the widtch of the histogram of each window is
        estimated to be $σ, which is smaller than the window distance $Δwindow.
        This may cause insufficient overlap between windows.
        Possible solutions:
        1. Use smaller force constant.
        2. Use more windows.
        3. Use umbrellia integration instead of WHAM for unbiasing.""")
    end
    xi = [bound[1] + (i-0.5)*Δwindow for i in 1:nw]
    return xi
end

function umbrellaSampling(temp::T1, nw::T2, nStep::T2, ks::T1, bound::Vector{T1},
    ωc::T1, chi::T1) where {T1<:Real, T2<:Integer}
    nSkip = 10
    t = temp / au2kelvin
    nCollected = floor(Int64, nStep/nSkip)
    xi = umbrellaSetup(t, nw, ks, bound)
    param, mol, bath, uTotal, cache = initialize(temp, freqCutoff, eta, ωc, chi)
    wham_prarm, wham_array, ui_array =  WHAM.setup(t, nw, bound, xi, ks/2.0, nBin=10*nw+1)

    rng = Random.seed!(114514+Threads.threadid())
    # rng = Random.seed!()
    cv = Vector{Float64}(undef, nCollected)
    for i in 1:nw
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        x0 = xi[i]
        mol.x .= -2.0
        mol.x[1] = x0
        mol.x[end] = 0.0
        Dynamics.velocitySampling!(mol, rng)
        Dynamics.velocitySampling!(bath, rng)
        bath.x = Random.randn(param.nBath)
        Dynamics.force!(mol, bath, uTotal, cache)
        for j in 1:5000
            Dynamics.velocityVelert!(mol, bath, param, uTotal, cache, ks, x0)
            if j % 25 == 0
                Dynamics.velocitySampling!(mol, rng)
                Dynamics.velocitySampling!(bath, rng)
            end
        end
        # println(output, "# ", v0)
        for j in 1:nCollected
            for k in 1:nSkip
                Dynamics.velocityVelert!(mol, bath, param, uTotal, cache, ks, x0)
            end
            cv[j] = mol.x[1]
            # if j % 25 == 0
                Dynamics.velocitySampling!(mol, rng)
                Dynamics.velocitySampling!(bath, rng)
            # end
            # println(j, " ", mol.v[1], " ", q[j+1], " ", 0.5*amu2au*mol.v[1]^2+pesMol(mol.x[1]))
            # println(j, " ", mol.x[1], " ", q[j+1])
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", uTotal(mol.x[1], mol.x[2]), " ", q[j+1])
        end
        # wham_array = WHAM.biasedDistibution(cv, i, wham_prarm, wham_array)
        println("Processing window number $i")
        ui_array.mean[i], ui_array.var[i] = WHAM.windowStats(cv)
        # close(output)
    end

    # xbin, pmf = @time WHAM.unbias(wham_prarm, wham_array)
    xbin, pmf = @time WHAM.integration(wham_prarm, ui_array)
    flnm = string("pmf_", ωc, "_", chi, "_", temp, "_", param.nMol, ".txt")
    open(flnm, "w") do io
        @printf(io, "# ω_c=%5.3f,χ=%6.4f \n", ωc, chi)
        @printf(io, "# Thread ID: %3d\n", Threads.threadid())
        @printf(io, "# Physical temperature: %6.1f\n", temp)
        @printf(io, "# Number of windows: %5d\n", nw)
        @printf(io, "# Number of bins: %5d\n", wham_prarm.nBin)
        @printf(io, "# Number of points per window: %11d\n", nStep)
        @printf(io, "# Convergence criteria: %7.2g\n", 1e-12)
        writedlm(io, [xbin pmf])
    end
    return xbin, pmf
end

function rate(out::IOStream, fs::AbstractVector{T}, tst::T, beta::T) where T <: AbstractFloat
    kappa = mean(fs[end-50:end])
    dev = varm(fs[end-50:end], kappa, corrected=false)    
    if dev >= 1e-6
        flag = "flagged"
    else
        flag = ""
    end
    kau = kappa * tst
    ksi = kau / au2fs * 1e15
    logkau = log(kau)
    logkEyring = log(kau*beta)
    logksi = log(ksi)
    temp = au2kelvin / beta
    invtemp = beta /au2kelvin
    @printf(out, "%7.5f %9.5f %9.5f %7.2f %9.5f %11.8g %11.8g %8.6g %11.8g %s\n",
        invtemp, logkau, logkEyring, temp, logksi, kau, ksi, kappa, tst, flag)
    return kau
end

const freqCutoff = 500.0 / au2wn
const eta = 4.0 * amu2au * freqCutoff
cd("..")
const pesMol, dipole = getPES()
# pesMol, dipole = getPES("pes_lowres.txt", "dm_lowres.txt")

function temperatureDependency()
    cd("tempDep")
    nTraj = 10000
    nStep = 3000
    temp = collect(233.0:15.0:413.0)
    # temp = [233.0]
    # omegac = vcat(collect(0.01:0.01:0.03), collect(0.1:0.1:0.3))
    omegac = [0.16]
    # chi = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    # iter = vcat([(0.16, 0.16i) for i in chi])
    # iter = reduce(vcat, [[(i,0.1i), (i,0.3i)] for i in omegac])
    iter = [(0.16, 0.0)]
    cd("tmp")
    computeKappa(300.0, 1, 1, omegac[1], omegac[1])
    cd("..")

    pmf = readdlm("pmf_0.2_0.0_300.0.txt", comments=true)
    # xbin, pmf = umbrellaSampling(temp[end], 100, 0.15, [-3.5, 3.5], 0.2, 0.0)
    Threads.@threads for i in iter
        ωc = i[1]
        χ = i[2]
        if length("$χ") >= 10
            χ -= eps(χ)
        end
        flnm = string("rate_", ωc, "_", χ, ".txt")
        output = open(flnm, "w")
        @printf(output, "# ω_c=%5.3f,χ=%6.4f \n", ωc, χ)
        @printf(output, "# Thread ID: %3d\n", Threads.threadid())
        @printf(output, "# Warning: check κ mannually if line ends with 'flagged'\n")
        @printf(output, "# 1/T    log k/au    log k/T     T      log k/s     k(au)       k(s^-1)      κ        TST(au)     Flag\n")
        @time for j in temp
            println("Currently running ωc = $ωc, χ = $χ, T = $j")
            fs = computeKappa(j, nTraj, nStep, ωc, χ)
            tst = WHAM.TSTRate(pmf[:, 1], pmf[:, 2], au2kelvin/j, amu2au)
            k = rate(output, fs, tst, au2kelvin/j)
        end
    end
end

# temperatureDependency()
cd("energy")
using Profile
function testKappa()
    @time computeKappa(300.0, 1, 1, 0.16, 0.0048)
    Profile.clear_malloc_data()
    @time computeKappa(300.0, 5000, 3000, 0.16, 0.0048)
end
function testPMF()
    @time umbrellaSampling(300.0, 100, 10, 0.15, [-3.5, 3.5], 0.16, 0.0)
    Profile.clear_malloc_data()
    @time umbrellaSampling(300.0, 100, 10000000, 0.15, [-3.5, 3.5], 0.16, 0.0)
end
testKappa()
# param, label, mass, bath = initialize(temp, freqCutoff, eta)
# const pesMol, dipole = getPES("pes_low.txt", "dm_low.txt")
# pesMol, dipole = getPES("pes_prx.txt", "dm_prx.txt")
# using Calculus: second_derivative
# using Optim, LineSearches
# 
# println(sqrt(-second_derivative(x -> pesMol(x), 0.0)/amu2au)*au2wn)
# result = optimize(x -> pesMol(x[1]), [-4.0], BFGS())
# x = Optim.minimizer(result)
# println(sqrt(second_derivative(x -> pesMol(x), x[1])/amu2au)*au2wn)
# 
# result = optimize(x -> pesMol(x[1]), [-1.0], BFGS())
# println(Optim.minimizer(result))

# chi = [0.0, 0.0002, 0.002, 0.02]
# omegac = collect(0.04:0.04:0.2)
# iter = vec([(i,j) for i in omegac, j in chi])
# run(1, omegac[1], chi[1])
# 
# chi = collect(0.001:0.001:0.019)
# chi = vcat(chi, collect(0.04:0.02:0.2))
# omegac = [0.2]
# iter = vcat(iter, vec([(i,j) for i in omegac, j in chi]))
# 
# chi = [0.02]
# omegac = collect(0.2:0.2:1.0)
# iter = vcat(iter, vec([(i,j) for i in omegac, j in chi]))
# @time Threads.@threads for i in iter 
#     run(nTraj, i[1], i[2]) 
# end

#  for i in [500]
#      temp = i / au2kelvin
#      @time kappa(temp, 1, 0.2, 0.2)
#      @time kappa(temp, 5000, 0.2, 0.2)
#  end

# chi = [0.2]
# omegac = vcat([0.001, 0.005, 0.01], collect(0.04:0.04:0.2), collect(0.4:0.2:1.0), collect(1.0:1.0:4.0))
# omegac = collect(1.0:3.0)
# iter = [(i,0.1i^1.5) for i in omegac]
# iter = [(i,0.05i) for i in omegac]
# iter = [(0.16,0.0064), (0.36,0.0216)]
# iter = vec([(i,j) for i in omegac, j in chi])
# @time Threads.@threads for i in iter 
#     flnm = string("fs_", i[1], "_", i[2], ".txt")
#     if isfile(flnm)
#         println(flnm, " already exists")
#         continue
#     end
#     kappa(temp, 5000, i[1], i[2]) 
# end
# pesMol, dipole = getPES()
# chi = 0.02
# wc = vcat([0.001, 0.005, 0.01, 0.05, 0.1], collect(0.11:0.01:0.2), collect(0.3:0.1:0.9), collect(1.0:10.0))
# 
# using Printf
# for i in wc
#     i /= au2ev
#     uTotal = constructPotential(pesMol, dipole, i, chi)
#     minLoc, minVal = Auxiliary.optPES(uTotal, [1.0, 1.0], Auxiliary.Optim.BFGS)
#     eTS = uTotal(0.0, 0.0)
#     ΔE = eTS - minVal
#     @printf("%9.7f %9.7f %9.7f %9.5f\n", minVal, eTS, ΔE, exp(-ΔE/300*au2kelvin))
# end
