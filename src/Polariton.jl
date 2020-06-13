# TODO: use ForwardDiff.gradient! instead of Calculus.gradient
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
    @. p.v += 0.5 * fc[1] / p.m * param.Δt 
    @. b.v += 0.5 * fc[2] / b.m * param.Δt 
    return p, b
end


function velocityVelert(param::Parameters, p::Particles1D, b::Bath1D,
    fc::Tuple, f::Function; cnstr=true)

    p, b = velocityUpdate(param, fc, p, b)
    @. p.x[2:end] += p.v[2:end] * param.Δt
    p.x[1] = 0.0
    @. b.x += b.v * param.Δt
    fc = force(f, p.x, b)
    mol, bath = velocityUpdate(param, fc, p, b)
    return p, b, fc
end

function velocityVelert(param::Parameters, p::Particles1D, b::Bath1D,
    fc::Tuple, f::Function)

    p, b = velocityUpdate(param, fc, p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    fc = force(f, p.x, b)
    mol, bath = velocityUpdate(param, fc, p, b)
    return p, b, fc
end

function velocityVelert(param::Parameters, p::Particles1D, b::Bath1D,
    fc::Tuple, f::Function, ks::T, x0::T) where T <: AbstractFloat

    p, b = velocityUpdate(param, fc, p, b)
    @. p.x += p.v * param.Δt
    @. b.x += b.v * param.Δt
    fc = force(f, p.x, b, ks, x0)
    mol, bath = velocityUpdate(param, fc, p, b)
    return p, b, fc
end

function force(f::Function, q, b::Bath1D)
    fp = f(q)
    fb = Vector{Float64}(undef, b.n)
    for i in 1:b.n
        tmp = b.c_mω2[i] * q[1] - b.x[i]
        fb[i] = b.mω2[i] * tmp
        fp[1] -= b.c[i] * tmp
    end
    return fp, fb
end

function force(f::Function, q::Vector{T}, b::Bath1D, ks::T, x0::T) where T <: AbstractFloat
    fp = f(q)
    fp[1] -= ks * (q[1]-x0)
    fb = Vector{Float64}(undef, b.n)
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
using StatProfilerHTML
using DelimitedFiles
using Interpolations
using Printf
using Random
using WHAM
using ..Dynamics
using ..Auxiliary
using .Auxiliary.Constants: au2wn, au2ev, au2kelvin, au2kcal, amu2au, au2fs
using ..CorrelationFunctions

const corr = CorrelationFunctions

function initialize(temp, freqCutoff, eta, ωc, chi)
    ωc /= au2ev
    param = Dynamics.Parameters(temp, 8.0, 2, 1, 15, 1, 1, 1)
    label = repeat(["mol"], param.nMol)
    mass = repeat([amu2au], param.nMol)
    ω = Vector{Float64}(undef, param.nBath)
    for i in 1:param.nBath
        ω[i] = -freqCutoff * log((i-0.5) / param.nBath)
    end
    c = ω * sqrt(2eta * amu2au * freqCutoff / param.nBath / pi)
    mω2 = ω.^2 .* amu2au
    c_mω2 = c ./ mω2
    bath = Dynamics.ClassicalBathMode(param.nBath, amu2au, ω, c, mω2, c_mω2,
        zeros(param.nBath), zeros(param.nBath))
    uTotal = constructPotential(pesMol, dipole, ωc, chi)
    if param.nParticle - param.nMol == 1 && chi != 0.0
        push!(label, "photon")
        push!(mass, 1.0)
        f = gradient(x -> -uTotal(x[1], x[2]))
        # tmp = Auxiliary.normalModeAnalysis(uTotal, zeros(2), omegaC)
        # axis = tmp[2][:, 1]
        # @. axis *= sqrt(mass)
        # println(axis, " ", Threads.threadid())
    else
        f = gradient(x -> -pesMol(x[1]))
    end
    mol = Dynamics.ClassicalParticle(label, mass, zeros(length(mass)),
        zeros(length(mass)))
    return param, mol, bath, f, uTotal
end

function getPES(pes="pes.txt", dm="dm.txt")
    potentialRaw = readdlm(pes)
    dipoleRaw = readdlm(dm)
    xrange = LinRange(potentialRaw[1, 1], potentialRaw[end, 1],
        length(potentialRaw[:, 1]))
    pesMol = CubicSplineInterpolation(xrange, potentialRaw[:, 2],
        extrapolation_bc=Line())
    xrange = LinRange(dipoleRaw[1, 1], dipoleRaw[end, 1],
        length(dipoleRaw[:, 1]))
    dipole = CubicSplineInterpolation(xrange, dipoleRaw[:, 2], extrapolation_bc=Line())
           return pesMol, dipole
end

function constructPotential(pesMol::T, dipole::T, omegaC::AbstractFloat,
    chi::AbstractFloat) where T <: AbstractExtrapolation
    # photon DOF potential
    pesPho(qPho) = 0.5 * omegaC^2 * qPho^2
    # light-mater interaction
    lightMatter(qMol, qPho) = sqrt(2*omegaC)*chi*dipole(qMol)*qPho +
        chi^2*dipole(qMol)^2/omegaC
    # total potential
    uTotal(qMol, qPho) = pesMol(qMol) + pesPho(qPho) + lightMatter(qMol, qPho)
    return uTotal
end

function kappa(temp, nTraj, ωc, chi)
    nSteps = 1500

    param, mol, bath, f, uTotal = initialize(temp, freqCutoff, eta, ωc, chi)

    fs0 = 0.0
    fs = zeros(nSteps+1)
    q = zeros(nSteps+1)
    x0 = zeros(length(mol.x))
    xb0 = Random.randn(bath.n)
    for i in 1:nTraj
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        mol.x = deepcopy(x0)
        mol.v = Dynamics.velocitySampling(param, mol)
        bath.v = Dynamics.velocitySampling(param, bath)
        bath.x = deepcopy(xb0)
        fc = Dynamics.force(f, mol.x, bath)
        for j in 1:1000
            mol, bath, fc = Dynamics.velocityVelert(param, mol, bath, fc, f,
                cnstr=true)
            if j % 50 == 0
                mol.v = Dynamics.velocitySampling(param, mol)
                bath.v = Dynamics.velocitySampling(param, bath)
            end
        end
        x0 = deepcopy(mol.x)
        xb0 = deepcopy(bath.x)
        v0 = mol.v[1] 
        q .= 0.0
        # println(output, "# ", v0)
        fs0 = corr.fluxSide(fs0, v0, v0)
        for j in 1:nSteps
            mol, bath, fc = Dynamics.velocityVelert(param, mol, bath, fc, f)
            q[j+1] = mol.x[1]
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
    printKappa(fs, ωc, chi, param.Δt)
end

function printKappa(fs, ωc, chi, dt)
    fs[1] = 1.0
    flnm = string("fs_", ωc, "_", chi, ".txt")
    fsOut = open(flnm, "w")
    @printf(fsOut, "# Thread ID %3d\n", Threads.threadid())
    @printf(fsOut, "# ω_c=%7.3e,χ=%6.4g \n", ωc, chi)
    for i in 1:length(fs)
        @printf(fsOut, "%5.2f %5.3g\n", (i-1)*dt*2.4189e-2, fs[i])
    end
    close(fsOut)
    println("Results written to file: $fsOut")
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
    xi = [ bound[1] + (i-0.5)*Δwindow for i in 1:nw]
    return xi
end

function umbrellaSampling(temp::T, nw::Integer, ks::T, bound::Vector{T}, ωc::T, chi::T
    ) where T <: AbstractFloat
    nSteps = 50000000
    bound = [-3.5, 3.5]
    xi = umbrellaSetup(temp, nw, ks, bound)
    param, mol, bath, f, uTotal = initialize(temp, freqCutoff, eta, ωc, chi)
    wham_prarm, wham_array =  WHAM.setup(temp, nw, bound, xi, ks/2.0, nBin=10*nw)

    cv = Vector{Float64}(undef, nSteps)
    for i in 1:nw
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        x0 = xi[i]
        mol.x .= 0.0 
        mol.v = Dynamics.velocitySampling(param, mol)
        bath.v = Dynamics.velocitySampling(param, bath)
        bath.x = Random.randn(bath.n)
        fc = Dynamics.force(f, mol.x, bath)
        for j in 1:1000
            mol, bath, fc = Dynamics.velocityVelert(param, mol, bath, fc, f,
                ks, x0)
            if j % 25 == 0
                mol.v = Dynamics.velocitySampling(param, mol)
                bath.v = Dynamics.velocitySampling(param, bath)
            end
        end
        # println(output, "# ", v0)
        for j in 1:nSteps
            mol, bath, fc = Dynamics.velocityVelert(param, mol, bath, fc, f,
                ks, x0)
            cv[j] = mol.x[1]
            if j % 25 == 0
                mol.v = Dynamics.velocitySampling(param, mol)
                bath.v = Dynamics.velocitySampling(param, bath)
            end
            # println(j, " ", mol.v[1], " ", q[j+1], " ", 0.5*amu2au*mol.v[1]^2+pesMol(mol.x[1]))
            # println(j, " ", mol.x[1], " ", q[j+1])
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", uTotal(mol.x[1], mol.x[2]), " ", q[j+1])
        end
        wham_array = WHAM.biasedDistibution(cv, i, wham_prarm, wham_array)
        # close(output)
    end

    xbin, pmf = @time WHAM.unbias(wham_prarm, wham_array)
    flnm = string("pmf_", ωc, "_", chi, ".txt")
    open(flnm, "w") do io
        @printf(io, "# ω_c=%5.3f,χ=%6.4f \n", ωc, chi)
        @printf(io, "# Thread ID: %3d\n", Threads.threadid())
        @printf(io, "# Number of windows: %5d\n", nw)
        @printf(io, "# Number of bins: %5d\n", wham_prarm.nBin)
        @printf(io, "# Number of points per window: %11d\n", nSteps)
        @printf(io, "# Convergence criteria: %7.2g\n", 1e-12)
        writedlm(io, [xbin pmf])
    end
    return xbin, pmf
end

Random.seed!(1234)
nTraj = 5000
temp = 300.0 / au2kelvin
freqCutoff = 500.0 / au2wn
eta = 4.0 * amu2au * freqCutoff
cd("..")
pesMol, dipole = getPES()
# param, label, mass, bath = initialize(temp, freqCutoff, eta)
# const pesMol, dipole = getPES("pes_low.txt", "dm_low.txt")
# pesMol, dipole = getPES("pes_prx.txt", "dm_prx.txt")
# cd("chi_wc")
cd("1D")
# xbin, pmf = umbrellaSampling(temp, 100, 0.08, [-3.5, 3.5], 0.2, 0.2)
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
omegac = vcat([0.001, 0.005, 0.01], collect(0.04:0.04:0.2), collect(0.4:0.2:1.0), collect(1.0:1.0:4.0))
# omegac = collect(1.0:3.0)
# iter = [(i,0.1i^1.5) for i in omegac]
iter = [(i,0.05i) for i in omegac]
# iter = [(0.16,0.0064), (0.36,0.0216)]
# iter = vec([(i,j) for i in omegac, j in chi])
@time Threads.@threads for i in iter 
    flnm = string("fs_", i[1], "_", i[2], ".txt")
    # if isfile(flnm)
    #     println(flnm, " already exists")
    #     continue
    # end
    kappa(temp, 5000, i[1], i[2]) 
end
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