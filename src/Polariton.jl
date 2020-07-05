module Auxiliary
"""
    Some physical constants useful to the simulation
"""

using Constants
using Calculus
using IntervalArithmetic, IntervalRootFinding

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
    function saddlePoints(f::Function, low::T, high::T) where T<:Real

Find the roots of a given univariate function f in a given range [low, high],
and return a sorted vector of roots via `IntervalRootFinding`. Note that only 
a pure Julia function will work with `IntervalRootFinding`.
"""
function saddlePoints(f::Function, low::T, high::T) where T<:Real
    result = roots(f, low..high)
    loc = Vector{Float64}(undef, 0)
    for i in result
        push!(loc, mid(i.interval))
    end
    return sort(loc)
end

end

include("CorrelationFunctions.jl")
include("Dynamics.jl")
include("Initialization.jl")
include("Fitting.jl")

using DelimitedFiles
using Printf
using Random
using WHAM
using Statistics: mean, varm
using ..Dynamics
using ..Auxiliary
using ..CorrelationFunctions

const corr = CorrelationFunctions

function computeKappa(nParticle::T2, temp::T1, nTraj::T2, nStep::T2, ωc::T1,
    chi::T1) where {T1<:Real, T2<:Integer}

    rng = Random.seed!(1233+Threads.threadid())
    param, mol, bath, forceEval!, cache, flnmID = initialize(nParticle, temp,
        ωc, chi)

    dir = string(param.nMol)
    # if ! isdir(dir)
    #     mkdir(dir)
    # end
    # cd(string(param.nMol))
    fs0 = 0.0
    fs = zeros(nStep)
    q = zeros(nStep)
    # e = zeros(nStep+1)
    # flnm0 = string("t0_", param.nMol, ".txt")
    # flnm10 = string("t10_", param.nMol, ".txt")
    # t0 = open(flnm0, "w")
    # t10 = open(flnm10, "w")
    Dynamics.velocitySampling!(mol, rng)
    Dynamics.velocitySampling!(bath, rng)
    Dynamics.force!(mol, bath, forceEval!)
    Dynamics.equilibration!(mol, bath, 8000, rng, param, forceEval!, cache)
    xm0 = copy(mol.x)
    xb0 = copy(bath.x)
    vm0 = copy(mol.v)
    vb0 = copy(bath.v)
    @inbounds for i in 1:nTraj
        # traj = string("traj_", param.nMol, "_test.txt")
        # output = open(traj, "w")
        copy!(mol.x, xm0)
        copy!(bath.x, xb0)
        copy!(mol.v, vm0)
        copy!(bath.v, vb0)
        Dynamics.equilibration!(mol, bath, 1000, rng, param, forceEval!, cache)
        copy!(xm0, mol.x)
        copy!(xb0, bath.x)
        copy!(vm0, mol.v)
        copy!(vb0, bath.v)
        Dynamics.velocitySampling!(mol, rng)
        Dynamics.velocitySampling!(bath, rng)
        v0 = mol.v[1] 
        # println(t0, v0)
        # e[1] += reactiveEnergy(mol)
        # println(output, "# ", v0)
        fs0 = corr.fluxSide(fs0, v0, v0)
        @inbounds for j in 1:nStep
            Dynamics.velocityVerlet!(mol, bath, param, forceEval!)
            q[j] = mol.x[1]
            # println(output, j, " ", mol.x[1])
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", mol.x[end])
            # println(output, j, " ", mean(@view mol.x[2:end-1]))
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", 918.0 * mol.v[1]^2)
            # e[j+1] += reactiveEnergy(mol)
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", mol.x[3], " ", mol.x[4])
            # if q[j+1] * q[j] < 0
            #     println(output, "# here")
            # end
        end
        # println(t10, mol.v[1])
        # close(output)
        # fs .+= q
        corr.fluxSide!(fs, v0, q)
    end
    # close(t0)
    # close(t10)
    # e /= nTraj + 0.0
    # open("check_energy.txt", "w") do io
    #     writedlm(io, e)
    # end

    return printKappa(fs, fs0, ωc, chi, temp, param)
end

function reactiveEnergy(p::Dynamics.ClassicalParticle)
    return pes(p.x[1]) + 918.0 * p.v[1]^2
end

function printKappa(fs::AbstractVector{T}, fs0::T, ωc::T, chi::T, temp::T,
    param::Dynamics.Parameters) where T <: AbstractFloat
    fs ./= fs0
    # @. fs /= 100.0
    flnm = string("fs_", ωc, "_", temp, "_", param.nMol, "_lf.txt")
    # flnm = string("fs_", ωc, "_", chi, "_", temp, "_", param.nMol, "_v0.txt")
    fsOut = open(flnm, "w")
    @printf(fsOut, "# Thread ID %3d\n", Threads.threadid())
    @printf(fsOut, "# ω_c=%7.3e,χ=%6.4g \n", ωc, chi)
    @printf(fsOut, "%5.2f %11.8g\n", 0.0, 1.0)
    for i in 1:length(fs)
        @printf(fsOut, "%5.2f %11.8g\n", (i-1)*param.Δt*2.4189e-2, fs[i])
    end
    close(fsOut)
    println("Results written to file: $flnm")
    # cd("..")
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
    param, mol, bath, forceEval!, cache = initialize(2, temp, freqCutoff, eta,
        ωc, chi)
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
        Dynamics.force!(mol, bath, forceEval!, cache)
        for j in 1:5000
            Dynamics.velocityVerlet!(mol, bath, param, forceEval!, cache, ks, x0)
            if j % 25 == 0
                Dynamics.velocitySampling!(mol, rng)
                Dynamics.velocitySampling!(bath, rng)
            end
        end
        # println(output, "# ", v0)
        for j in 1:nCollected
            for k in 1:nSkip
                Dynamics.velocityVerlet!(mol, bath, param, forceEval!, cache, ks, x0)
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

cd("..")
# const dipoleCoeff, v0, pesCoeff = getCoefficients(coeffFile="coefficients.jld")
# const dipoleCoeff, v0, pesCoeff = getCoefficients()

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
    computeKappa(2, 300.0, 1, 1, omegac[1], omegac[1])
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
            fs = computeKappa(2, j, nTraj, nStep, ωc, χ)
            tst = WHAM.TSTRate(pmf[:, 1], pmf[:, 2], au2kelvin/j, amu2au)
            k = rate(output, fs, tst, au2kelvin/j)
        end
    end
end

# temperatureDependency()
cd("chk")
function testKappa()
    if length(ARGS) != 0
        np = parse(Int64, ARGS[1])
    else
        np = 2
    end
    chi = 0.01 # / sqrt(np)
    @time computeKappa(np, 300.0, 1, 1, 0.18, chi)
    @time computeKappa(np, 300.0, 1000, 3000, 0.18, chi)
end
function testPMF()
    @time umbrellaSampling(300.0, 100, 10, 0.15, [-3.5, 3.5], 0.16, 0.0)
    Profile.clear_malloc_data()
    @time umbrellaSampling(300.0, 100, 10000000, 0.15, [-3.5, 3.5], 0.16, 0.0)
end
testKappa()
# param, label, mass, bath = initialize(temp, freqCutoff, eta)
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
