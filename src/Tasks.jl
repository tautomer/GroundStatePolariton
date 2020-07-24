include("CorrelationFunctions.jl")
include("Dynamics.jl")
include("Initialization.jl")
include("Fitting.jl")

using Parameters
using DelimitedFiles
using Printf
using Random
using WHAM
using Statistics: mean, varm
using ..Dynamics
using ..CorrelationFunctions
const corr = CorrelationFunctions

@with_kw mutable struct InputValues{T1<:Integer, T2<:AbstractFloat, T3<:Symbol}
    np::T1
    ntraj::T1
    nstep::T1
    temp::T2
    ωc::T2
    χ::T2
    dynamics::T3
    model::T3
    alignment::T3
end

function computeKappa(input::InputValues)

    @unpack np, ntraj, nstep, = input
    rng, param, mol, bath, forceEval!, cache, flnmID = initialize(np,
        input.temp, input.ωc, input.χ, dynamics=input.dynamics,
        model=input.model, alignment=input.alignment)
    
    if param.nMol == 1
        alignment = Val(:ordered)
    else
        alignment = Val(input.alignment)
    end

    fs0 = 0.0
    fs = zeros(nstep)
    q = zeros(nstep)
    Dynamics.velocitySampling!(mol, bath, rng)
    Dynamics.force!(mol, bath, forceEval!, alignment)
    Dynamics.equilibration!(mol, bath, 4000, rng, param, forceEval!, cache, alignment)
    if input.dynamics == :langevin
        savedArrays = (copy(mol.x), copy(mol.f), copy(mol.v))
    else
        savedArrays = (copy(mol.x), copy(mol.f), copy(mol.v), copy(bath.x),
            copy(bath.f), copy(bath.v))
    end
    @inbounds for i in 1:ntraj
        # traj = string("traj_", i, "_model.txt")
        # traj = string("traj_", i, "_model_large.txt")
        # output = open(traj, "w")
        if i % 100 == 0
            println("Running trajcetory $i")
        end
        if alignment == Val(:disordered)
            getRandomAngles!(mol.cosθ, rng)
            if input.model == :normalModes
                mol.sumCosθ = sum(mol.cosθ)
            end
        end
        Dynamics.copyArrays!(savedArrays, mol, bath)
        Dynamics.equilibration!(mol, bath, 3000, rng, param, forceEval!, cache,
            alignment)
        Dynamics.copyArrays!(mol, bath, savedArrays)
        Dynamics.velocitySampling!(mol, bath, rng)
        v0 = mol.v[1] 
        # println(t0, v0)
        # e[1] += reactiveEnergy(mol)
        # println(output, "# ", v0)
        fs0 = corr.fluxSide(fs0, v0, v0)
        @inbounds for j in 1:nstep
            Dynamics.velocityVerlet!(mol, bath, param, rng, forceEval!, cache,
                alignment)
            q[j] = mol.x[1]
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", mol.x[3])
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
    # # close(t0)
    # # close(t10)
    # # e /= nTraj + 0.0
    # # open("check_energy.txt", "w") do io
    # #     writedlm(io, e)
    # # end
    corr.normalize!(fs, fs0)

    return printKappa(fs, flnmID, param.Δt, flag="new_system")
end

function reactiveEnergy(p::Dynamics.FullSystemParticle)
    return pes(p.x[1]) + 918.0 * p.v[1]^2
end

function printKappa(fs::AbstractVector{T}, flnmID::S, dt::T; flag::S=""
    ) where {T<:AbstractFloat, S<:String}
    ωc = split(flnmID, "_")[1]
    χ = split(flnmID, "_")[2]
    if isempty(flag)
        flnm = string("fs_", flnmID, ".txt")
    else
        flnm = string("fs_", flnmID, "_", flag, ".txt")
    end
    fsOut = open(flnm, "w")
    @printf(fsOut, "# Thread ID %3d\n", Threads.threadid())
    @printf(fsOut, "# ω_c=%s,χ=%s \n", ωc, χ)
    @printf(fsOut, "%5.2f %11.8g\n", 0.0, 1.0)
    @inbounds for i in eachindex(fs)
        @printf(fsOut, "%5.2f %11.8g\n", i*dt*au2fs, fs[i])
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