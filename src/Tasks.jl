include("CorrelationFunctions.jl")
include("Dynamics.jl")
include("Initialization.jl")
include("Fitting.jl")

using Parameters
using DelimitedFiles
# using HDF5
using Printf
using Random
using WHAM
using Statistics: mean, varm
using ..Dynamics
using ..CorrelationFunctions
const corr = CorrelationFunctions

@with_kw mutable struct KappaInput{T1<:Integer, T2<:AbstractFloat, T3<:Symbol}
    np::T1
    nb::T1
    ntraj::T1
    nstep::T1
    constrained::T1
    temp::T2
    ωc::T2
    χ::T2
    dynamics::T3
    model::T3
    alignment::T3
    barriers::T3
end

@with_kw mutable struct UmbrellaInput{T1<:Integer, T2<:AbstractFloat, T3<:Symbol}
    np::T1
    nb::T1
    nw::T1
    nstep::T1
    constrained::T1
    temp::T2
    ks::T2
    bound::Vector{T2}
    ωc::T2
    χ::T2
    unbias::T3
    barriers::T3
end

function computeKappa(input::KappaInput)

    @unpack np, nb, ntraj, nstep, = input
    if nb > 1 && input.dynamics == :langevin
        println("White-noise Langevin dynamics does not work with RPMD")
        input.dynamics = :systemBath
    end

    rng, param, mol, bath, forceEval!, cache, flnmID = initialize(np, nb,
        input.temp, input.ωc, input.χ, constrained=input.constrained,
        barriers=input.barriers, dynamics=input.dynamics, model=input.model, 
        alignment=input.alignment)
    # dir = string(input.ωc, "_", input.χ)
    # if ! isdir(dir)
    #     mkdir(dir)
    # end
    # cd(dir)
    if param.nMol == 1
        alignment = Val(:ordered)
    else
        alignment = Val(input.alignment)
    end

    fs0 = 0.0
    fs = zeros(Float64, nstep)
    q = Vector{Float64}(undef, nstep)
    # vb = copy(mol.v)
    Dynamics.velocitySampling!(mol, bath, rng)
    Dynamics.force!(mol, bath, forceEval!, alignment)
    Dynamics.equilibration!(mol, bath, 4000, rng, param, forceEval!, cache,
        alignment, Val(:cnstr))
    if input.dynamics == :langevin
        savedArrays = (copy(mol.x), copy(mol.f), copy(mol.v))
    else
        savedArrays = (copy(mol.x), copy(mol.f), copy(mol.v), copy(bath.x),
            copy(bath.f), copy(bath.v))
    end
    skip = ntraj / 20
    # side = open("train_side.txt", "w")
    # init = h5open("training_data.h5", "w")
    # xvf = Vector{Float64}(undef, 249)
    @inbounds for i in 1:ntraj
        if i % skip == 0
            println("Running trajcetory ", i)
        end
        if alignment == Val(:disordered)
            getRandomAngles!(mol.cosθ, rng)
            if input.model == :normalModes
                mol.sumCosθ = sum(mol.cosθ)
            end
        end
        Dynamics.copyArrays!(savedArrays, mol, bath)
        Dynamics.equilibration!(mol, bath, 1000, rng, param, forceEval!, cache,
            alignment, Val(:cnstr))
        Dynamics.copyArrays!(mol, bath, savedArrays)
        Dynamics.velocitySampling!(mol, bath, rng)
        # xvf[1:3] .= mol.x
        # xvf[4:6] .= mol.v
        # xvf[7:9] .= mol.f
        # xvf[10:89] .= bath.x
        # xvf[90:169] .= bath.v
        # xvf[170:249] .= bath.f
        # write(init, string("molx", i), xvf)
            # writedlm(io, [mol.x mol.v mol.f])
            # writedlm(io, [bath.x bath.v bath.f])
        # copy!(vb, mol.v)
        # for k in 1:2
        # mol.x .= 0.0
        # @. mol.v = vb * (-1)^k
        v0 = corr.getCentroid(mol.v)
        # println(mol.x[2])
        fs0 = corr.fluxSide(fs0, v0, v0)
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        # println(output, 0, " ", mol.x[1], " ", mol.x[2], " ", mol.x[3],
        #     " ", cbo(mol.x, input.constrained, input.ωc/au2ev, input.χ/au2ev))
        # println(output, "# ", v0)
        # println(output, 0, " ", mol.x[1], " ", mol.x[2], " ", mol.x[3], " ")
        @inbounds for j in 1:nstep
            # Dynamics.velocityVerlet!(mol, bath, param, rng, forceEval!, cache,
            #     alignment)
            Dynamics.velocityVerlet!(mol, bath, param, forceEval!, alignment)
            q[j] = corr.getCentroid(mol.x)
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", mol.x[3],
            #     " ", cbo(mol.x, input.constrained, input.ωc/au2ev, input.χ/au2ev))
            # if q[j] * q[j-1] < 0
            #     println(output, "# here")
            # end
        end
        # h5open(string("traj_", i, ".txt"), "w") do io
        #     write(io, "q", q)
        # end
        # println(side, corr.heaviside(q[end]))
        # close(output)
        corr.fluxSide!(fs, v0, q)
    end
    # close(init)
    # close(side)
    corr.normalize!(fs, fs0)

    return printKappa(fs, flnmID, param.Δt)
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

function umbrellaSetup(temp::T, nw::Integer, ks::T, bound::Vector{T},
    method::Symbol) where T<:AbstractFloat
    sort!(bound)
    Δwindow = (bound[2]-bound[1]) / nw
    σ = sqrt(temp/ks)
    if method === :WHAM
        if σ < Δwindow
            println("""Warning: the widtch of the histogram of each window is
            estimated to be $σ, which is smaller than the window distance $Δwindow.
            This may cause insufficient overlap between windows.
            Possible solutions:
            1. Use smaller force constant.
            2. Use more windows.
            3. Use umbrella integration instead of WHAM for unbiasing.""")
        end
    end
    xi = [bound[1] + (i-0.5)*Δwindow for i in 1:nw]
    return xi
end

function umbrellaSampling(input::UmbrellaInput)

    @unpack np, nb, nw, nstep, ks, bound, = input

    nskip = 20
    nCollected = floor(Int64, nstep/nskip)
    temp = input.temp / au2kelvin
    xi = umbrellaSetup(temp, nw, ks, bound, input.unbias)
    rng, param, mol, bath, forceEval!, cache, flnmID = initialize(np, nb,
        input.temp, input.ωc, input.χ, ks=ks, constrained=input.constrained,
        barriers=input.barriers, dynamics=:systemBath, model=:fullSystem)
    us_param, pmf_array =  WHAM.setup(temp, nw, bound, xi, ks/2.0, nBin=10*nw+1,
        method=input.unbias)

    alignment = Val(:ordered)
    if param.beadMol > 1
        xi .*= sqrt(param.beadMol)
    end
    cv = Vector{Float64}(undef, nCollected)
    for i in 1:nw
        # traj = string("traj_", i, ".txt")
        # output = open(traj, "w")
        mol.xi = xi[i]
        @views mol.x[1:param.beadMol] .= us_param.windowCenter[i]
        @views mol.x[end-param.beadMol+1:end] .= 0.0
        Dynamics.velocitySampling!(mol, bath, rng)
        Dynamics.force!(mol, bath, forceEval!, alignment)
        Dynamics.equilibration!(mol, bath, 10000, rng, param, forceEval!, cache,
            alignment, Val(:restr))
        # println(output, "# ", v0)
        println("Running window number $i") 
        for j in 1:nCollected
            Dynamics.equilibration!(mol, bath, nskip, rng, param, forceEval!,
                cache, alignment, Val(:restr))
            cv[j] = WHAM.cv(mol.x)
            # println(j, " ", mol.v[1], " ", q[j+1], " ", 0.5*amu2au*mol.v[1]^2+pesMol(mol.x[1]))
            # println(output, j, " ", cv[j])
            # println(output, j, " ", mol.x[1], " ", mol.x[2], " ", uTotal(mol.x[1], mol.x[2]), " ", q[j+1])
        end
        WHAM.biased!(pmf_array, us_param, cv, i)
        # writedlm(output, [us_param.binCenter pmf_array.vBiased[:, i]])
        # close(output)
    end

    xbin, pmf = WHAM.unbias(pmf_array, us_param)
    flnm = string("pmf_", flnmID, "_more_points.txt")
    open(flnm, "w") do io
        @printf(io, "# ω_c=%5.3f,χ=%6.4f \n", input.ωc, input.χ)
        @printf(io, "# Thread ID: %3d\n", Threads.threadid())
        @printf(io, "# Physical temperature: %6.1f\n", input.temp)
        @printf(io, "# Number of windows: %5d\n", nw)
        @printf(io, "# Number of bins: %5d\n", us_param.nBin)
        @printf(io, "# Number of points per window: %11d\n", nstep)
        @printf(io, "# Convergence criteria: %7.2g\n", 1e-12)
        writedlm(io, [xbin pmf])
    end
    return xbin, pmf
end

function rate(out::IOStream, fs::AbstractVector{T}, tst::T, beta::T) where T<:AbstractFloat
    k = 3.4633383e-22
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
        invtemp, logkau, logkEyring, temp, logksi, kau, kau/k, kappa, tst, flag)
    return kau
end
