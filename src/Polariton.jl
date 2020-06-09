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
    mat_og = deepcopy(mat)
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
using Calculus
using LinearAlgebra: dot
using Dierckx
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
    c::Vector{Float64}
    ω::Vector{Float64}
    mω2::Vector{Float64}
    c_mω2::Vector{Float64}
    x::Array{Float64, 2}
    v::Array{Float64, 2}
end

function velocitySampling(param::Parameters, p::Particles1D)
    tmp = @. sqrt(param.temperature / p.m)
    return Random.randn(length(p.v)) .* tmp
end

function velocitySampling(param::Parameters, p::ParticlesND)
    n = param.beadMol
    v = Random.randn(n) * sqrt(n * param.temperature / p.m)
    return v
end

function velocityUpdate(param::Parameters, fc::Tuple, p::Particles1D, b::Bath1D)
    dt = param.Δt
    accdt1 = Vector{Float64}(undef, length(fc[1]))
    accdt2 = Vector{Float64}(undef, length(fc[2]))
    @. accdt1 = 0.5 * fc[1] / p.m * dt
    @. accdt2 = 0.5 * fc[2] / b.m * dt
    # println(accdt2)
    @. p.v += accdt1
    @. b.v += accdt2
    return p, b
end


function positionUpdate(param::Parameters, p::Particles1D, b::Bath1D; cnstr=true)
    @. p.x += p.v * param.Δt
    p.x[1] = 0.0
    @. b.x += b.v * param.Δt
    return p, b
end

function positionUpdate(param::Parameters, p::Particles1D, b::Bath1D)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    return p, b
end

function force(f::Function, q, b::Bath1D)
    fp = f(q)
    fb = Vector{Float64}(undef, b.n)
    # tmp = Vector{Float64}(undef, b.n)
    # @. tmp = b.c_mω2 * b.x - q[1]
    # @. fb = b.mω2 * tmp
    # fp[1] -= sum(b.c .* tmp)
    for i in 1:b.n
        tmp = b.c_mω2[i] * q[1] - b.x[i]
        fb[i] = b.mω2[i] * tmp
        fp[1] -= b.c[i] * tmp
    end
    return fp, fb
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

function heaviSide(x)
    if x >= 0
        h = 1
    else
        h = 0
    end
    return h
end

function fluxSide(corrFS, v0, q::Float64)
    side = heaviSide(q)
    corrFS += v0 * side
    return corrFS
end

function fluxSide(corrFS, v0, q::Array)
    side = heaviSide.(q)
    corrFS += v0 * side
    return corrFS
end

end

using Calculus: gradient
using DelimitedFiles
using Dierckx
using Printf
using Random
using ..Dynamics
using ..Auxiliary
using .Auxiliary.Constants: au2wn, au2ev, au2kelvin, au2kcal, amu2au, au2fs
using ..CorrelationFunctions

const corr = CorrelationFunctions

function getPES(pes="pes.txt", dm="dm.txt")
    potentialRaw = readdlm(pes)
    dipoleRaw = readdlm(dm)
    pesMol = Spline1D(potentialRaw[:, 1], potentialRaw[:, 2])
    dipole = Spline1D(dipoleRaw[:, 1], dipoleRaw[:, 2])
    return pesMol, dipole
end

function constructPotential(pesMol::Spline1D, dipole::Spline1D,
    omegaC::AbstractFloat, chi::AbstractFloat)
    # photon DOF potential
    pesPho(qPho) = 0.5 * omegaC^2 * qPho^2
    # light-mater interaction
    lightMatter(qMol, qPho) = sqrt(2*omegaC)*chi*dipole(qMol)*qPho +
        chi^2*dipole(qMol)^2/omegaC
    # total potential
    uTotal(qMol, qPho) = pesMol(qMol) + pesPho(qPho) + lightMatter(qMol, qPho)
    return uTotal
end

function run(nTraj, omegaC, chi)
    Random.seed!(1234)
    nSteps = 1500
    temp = 300.0 / au2kelvin
    ωc = omegaC
    omegaC /= au2ev
    freqCutoff = 500.0 / au2wn
    eta = 4.0 * amu2au * freqCutoff

    uTotal = constructPotential(pesMol, dipole, omegaC, chi)
    param = Dynamics.Parameters(temp, 8.0, 2, 1, 15, 1, 1, 1)
    label = repeat(["mol"], param.nMol)
    mass = repeat([amu2au], param.nMol)
    if param.nParticle - param.nMol == 1 && chi != 0.0
        push!(label, "photon")
        push!(mass, 1.0)
        f = gradient(x -> -uTotal(x[1], x[2]))
        axis = [1.0, 0.0]
        # tmp = Auxiliary.normalModeAnalysis(uTotal, zeros(2), omegaC)
        # axis = tmp[2][:, 1]
        # @. axis *= sqrt(mass)
        # println(axis, " ", Threads.threadid())
    else
        f = gradient(x -> -pesMol(x[1]))
        axis = zeros(2)
    end
    mol = Dynamics.ClassicalParticle(label, mass, zeros(length(mass)),
        zeros(length(mass)))
    ω = Vector{Float64}(undef, param.nBath)
    for i in 1:param.nBath
        ω[i] = -freqCutoff * log((i-0.5) / param.nBath)
    end
    c = ω * sqrt(2eta * amu2au * freqCutoff / param.nBath / pi)
    mω2 = ω.^2 .* amu2au
    c_mω2 = c ./ mω2
    bath = Dynamics.ClassicalBathMode(param.nBath, amu2au, ω, c, mω2, c_mω2,
        zeros(param.nBath), zeros(param.nBath))

    fs0 = 0.0
    fs = zeros(nSteps+1)
    q = zeros(nSteps+1)
    x0 = zeros(length(mass))
    xb0 = Random.randn(bath.n)
    for i in 1:nTraj
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        mol.x = deepcopy(x0)
        mol.v = Dynamics.velocitySampling(param, mol)
        bath.v = Dynamics.velocitySampling(param, bath)
        bath.x = deepcopy(xb0)
        fc = Dynamics.force(f, mol.x, bath)
        for i in 1:1000
            mol, bath = Dynamics.velocityUpdate(param, fc, mol, bath)
            mol, bath = Dynamics.positionUpdate(param, mol, bath, cnstr=true)
            fc = Dynamics.force(f, mol.x, bath)
            mol, bath = Dynamics.velocityUpdate(param, fc, mol, bath)
            if i % 50 == 0
                mol.v = Dynamics.velocitySampling(param, mol)
                bath.v = Dynamics.velocitySampling(param, bath)
            end
        end
        x0 = deepcopy(mol.x)
        xb0 = deepcopy(bath.x)
        v0 = mol.v[1] # corr.projection(mol.v, axis)
        q .= 0.0
        # println(output, "# ", v0)
        fs0 = corr.fluxSide(fs0, v0, v0)
        for j in 1:nSteps
            mol, bath = Dynamics.velocityUpdate(param, fc, mol, bath)
            mol, bath = Dynamics.positionUpdate(param, mol, bath)
            fc = Dynamics.force(f, mol.x, bath)
            mol, bath = Dynamics.velocityUpdate(param, fc, mol, bath)
            q[j+1] = mol.x[1] # corr.projection(mol.x, axis)
            # println(j, " ", mol.v[1], " ", q[j+1], " ", 0.5*amu2au*mol.v[1]^2+pesMol(mol.x[1]))
            # println(j, " ", mol.x[1], " ", q[j+1])
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", uTotal(mol.x[1], mol.x[2]), " ", q[j+1])
            # if q[j+1] * q[j] < 0
            #     println(output, "# here")
            # end
        end
        # close(output)
        fs = corr.fluxSide(fs, v0, q)
    end

    fs /= fs0
    fs[1] = 1.0
    flnm = string("fs_", ωc, "_", chi, ".txt")
    fsOut = open(flnm, "w")
    @printf(fsOut, "# Thread ID %3d\n", Threads.threadid())
    @printf(fsOut, "# ω_c=%5.3f,χ=%6.4f \n", ωc, chi)
    for i in 1:nSteps+1
        @printf(fsOut, "%5.2f %5.3f\n", i*param.Δt*2.4189e-2, fs[i])
    end
    close(fsOut)
end

nTraj = 5000
cd("..")
pesMol, dipole = getPES()
# const pesMol, dipole = getPES("pes_low.txt", "dm_low.txt")
# pesMol, dipole = getPES("pes_prx.txt", "dm_prx.txt")
# cd("wc_chi")
# cd("chk")
cd("1D")
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

chi = [0.0, 0.0002, 0.002, 0.02]
omegac = collect(0.04:0.04:0.2)
iter = vec([(i,j) for i in omegac, j in chi])
run(1, omegac[1], chi[1])

chi = collect(0.001:0.001:0.019)
chi = vcat(chi, collect(0.04:0.02:0.2))
omegac = [0.2]
iter = vcat(iter, vec([(i,j) for i in omegac, j in chi]))

chi = [0.02]
omegac = collect(0.2:0.2:1.0)
iter = vcat(iter, vec([(i,j) for i in omegac, j in chi]))
@time Threads.@threads for i in iter 
    run(nTraj, i[1], i[2]) 
end

# run(1, 0.2, 0.2)
# @time run(5000, 0.2, 0.2)

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