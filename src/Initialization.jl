include("FittedParameters.jl")
using Constants
using Random
using LinearAlgebra: dot
using Interpolations: LinearInterpolation, Line
using FiniteDiff: finite_difference_gradient!, GradientCache
push!(LOAD_PATH, pwd())
if isdefined(@__MODULE__,:LanguageServer)
    include("./RingPolymer.jl")
    using ..RingPolymer
else
    using RingPolymer
end

const freqCutoff = ω0[1]
const eta = 800.0 / au2wn / 2.0 * amu2au
# const eta = 0.5 * amu2au * freqCutoff
const gamma = 200 / au2wn # 400 cm^-1

"""
    function initialize(nParticle::T1, temp::T2, ωc::T2, chi::T2) where {T1<:Integer, T2<:Real}

Initialize most of the values, parameters and structs for the dynamics.
"""
# TODO better and more flexible way to handle input values
function initialize(nParticle::T1, nb::T1, temp::T2, ωc::T2, chi::T2;
    constrained::T1=1, ks::T2=0.0, barriers::T3=:oneBarrier,
    dynamics::T3=:langevin, model::T3=:normalModes, alignment::T3=:ordered
    ) where {T1<:Integer, T2<:Float64, T3<:Symbol}

    nPhoton = 1
    nMolecule = nParticle - nPhoton
    # number of bath modes per molecule! total number of bath mdoes = nMolecule * nBath
    nBath = 40
    dt = 4.0
    ν = 0.1
    τ = floor(Int64, 1/ν)
    # collisionFrequency 
    z = ν * dt

    # check if the number of molecules and photons make sense
    nParticle, nMolecule, label, mass = checkParticleNumbers(nParticle,
        nMolecule, chi, model)
    # total number of bath modes
    nBathTotal = nMolecule * nBath

    # the suffix for all output file names to identify the setup and avoid overwritting
    flnmID = string(ωc, "_", chi, "_", temp, "_", nb)
    # convert values to au, so we can keep more human-friendly values outside
    ωc /= au2ev
    chi /= au2ev
    temp /= au2kelvin
    # println("+: ", λ₊^2 * nMolecule * μeq * dμ0 / sqrt(amu2au) / ω₊, " ", ω₊)
    # println("-: ", λ₋^2 * nMolecule * μeq * dμ0 / sqrt(amu2au) / ω₋, " ", ω₋)

    rng = Random.seed!(1234)
    angles, x0 = initialPositions(nParticle, nMolecule, nb, constrained, rng,
        ωc, chi, barriers, alignment) 

    if dynamics == :systemBath
        bath, cache = buildBath(nBath, nMolecule, nb, x0, temp, dt)
    else
        # for Langevin dynamics
        bath, cache = langevinParameters(nMolecule, temp, dt, model, mass)
    end
    if model == :normalModes
        q₊, q₋, forceEvaluation! = reducedModelSetup(ωc, chi, sumCosθ)
        param = Dynamics.ReducedModelParameters(temp, dt, nMolecule)
        mol = Dynamics.ReducedModelParticle(label, sqrt(temp), dt/2.0,
            [0.0, q₊, q₋], zeros(3), zeros(3), angles, sumCosθ)
        return rng, param, mol, bath, forceEvaluation!, cache, flnmID
    end

    # angles for the dipole moment
    param = Dynamics.Parameters(temp, dt, z, τ, nParticle, nMolecule,
        nBathTotal, nb, nb, 1)
    if nb == 1
        σv = sqrt.(temp ./ mass)
        mol = Dynamics.FullSystemParticle(nMolecule, label, mass, σv, x0, ks,
            0.0, similar(mass), param.Δt./(2*mass), similar(mass), angles)
    else
        rpmd = ringPolymerSetup(nb, dt, temp)
        σv = sqrt.(temp * nb .* mass)
        halfdt = param.Δt / 2
        mol = Dynamics.RPMDParticle(nMolecule, nb, label, mass, σv, x0, ks*halfdt,
            0.0, similar(x0), similar(x0), halfdt, similar(x0), similar(x0),
            angles, rpmd.tnm, rpmd.tnmi, rpmd.freerp)
    end
    # obatin the gradient of the corresponding potential
    forceEvaluation! = constructForce(nParticle, nMolecule, constrained, ωc,
        chi, alignment, barriers)
    return rng, param, mol, bath, forceEvaluation!, cache, flnmID
end

"""
    function initialPositions(nParticle::T1, nMolecule::T1, nb::T1,
        constrained::T1, rng::AbstractRNG, ωc::T2, χ::T2, barriers::T3,
        alignment::T3) where {T1<:Integer, T2<:Float64, T3<:Symbol}

Compute the initial positions for R's and q_c.
"""
function initialPositions(nParticle::T1, nMolecule::T1, nb::T1,
    constrained::T1, rng::AbstractRNG, ωc::T2, χ::T2, barriers::T3,
    alignment::T3) where {T1<:Integer, T2<:Float64, T3<:Symbol}

    couple = sqrt(2/ωc^3) * χ
    # deal with ordered/disordered systems in many molecule case
    # TODO: is this still necessary?
    if alignment == :ordered
        angles = ones(nMolecule)
        sumCosθ = convert(Float64, nMolecule)
    else
        angles = Vector{Float64}(undef, nMolecule)
        getRandomAngles!(angles, rng)
        sumCosθ = sum(angles)
    end
    if barriers == :twoBarriers
        qPhoton = couple * -μeq[3-constrained] * (sumCosθ-1)
        x0 = [0.0, xeq[3-constrained], qPhoton]
    else
        if nMolecule > 1
            # x0[end] = couple * -μeq * (sumCosθ-1)
            qPhoton = couple * -μeq[1] * (sumCosθ-1)
        else
            qPhoton = 0.0
        end

        if nb == 1
            x0 = repeat([xeq[1]], nParticle)
            x0[1] = 0.0
            x0[end] = qPhoton
        else
            tmp = (rand(nb) .- 0.5) ./ sqrt(temp)
            x0 = repeat(tmp, 1, nParticle) ./ sqrt(amu2au)
            @views x0[:, 2:nMolecule] .+= xeq[1]
            @views x0[:, end] = tmp
        end
    end

    return angles, x0
end

"""
    function computeBathParameters(nBath::T) where T<:Integer

Compute bath parameters including coupling strength `cᵢ`, frequencies `ωᵢ` and
`mωᵢ^2` and `cᵢ/mωᵢ^2`. They will be used to compute forces.
"""
function computeBathParameters(nBath::T) where T<:Integer
    nb = convert(Float64, nBath)
    ω = Vector{Float64}(undef, nBath)
    mω2 = similar(ω)
    c = similar(ω)
    c_mω2 = similar(ω)
    tmp = sqrt(eta * amu2au * freqCutoff / nb)
    # tmp = sqrt(2eta * amu2au * freqCutoff / nb / pi)
    w0 = freqCutoff / nb * (1.0 - exp(-3.0))
    @inbounds @simd for i in eachindex(1:nBath)
        # bath mode frequencies
        # ω[i] = -freqCutoff * log((i-0.5) / nb)
        ω[i] = -freqCutoff * log(1 - i * w0 / freqCutoff)
        # bath force constants
        mω2[i] = ω[i]^2 * amu2au
        # bath coupling strength
        # c[i] = tmp * ω[i]
        c[i] = sqrt(2 * eta/pi * w0 * amu2au) * ω[i]
        # c/mω^2, for force evaluation
        c_mω2[i] = c[i] / mω2[i]
    end
    return ω, mω2, c, c_mω2 
end

function langevinParameters(nMolecule::Integer, temp::T, dt::T, model::Symbol,
    mass::AbstractVector{T}) where T<:Real

    # temporary variable for 0.5 * dt^2
    halfΔt2 = 0.5 * dt^2
    # 0.5 * dt^2 * gamma for position update
    halfΔt2γ = halfΔt2 * gamma
    # dt * gamma for velocity update
    dtγ = dt * gamma
    if model == :normalModes
        # sigma for the random force. Note dt is included
        σ = sqrt(2gamma * temp * dt)
        # 0.5 * dt^2 / m for position update
        dt2by2m = halfΔt2
    else
        σ = sqrt(2gamma * temp * dt / amu2au)
        dt2by2m = halfΔt2 ./ mass
    end
    # 0.5 * dt * sigma for the random force in position update
    dtσ = 0.5 * dt * σ
    # call it bath to keep it consistent with the name of the system-bath struct
    # last two fields are dummy, just to keep the main function functioning
    # as saving bath coordinates is necessary for system-bath model
    # TODO properly split system-bath model and langevin dynamics. A disptach might be necessary.
    if model == :normalModes
        bath = Dynamics.LangevinModes(gamma, σ, halfΔt2γ, dtγ, dtσ, dt2by2m,
            [1.0], [1.0])
        cache = Dynamics.LangevinCache(Vector{Float64}(undef, 3),
            Vector{Float64}(undef, 6), similar(mass), similar(mass))
    else
        bath = Dynamics.LangevinFull(gamma, σ, halfΔt2γ, dtγ, dtσ, dt2by2m,
            [1.0], [1.0])
        cache = Dynamics.LangevinCache(Vector{Float64}(undef, 3),
            Vector{Float64}(undef, 3*nMolecule),similar(mass),
            similar(mass))
    end
    return bath, cache
end

function reducedModelSetup(ωc::T, χ::T, sumCosθ::T) where T<:Real

    massWeight = sqrt(amu2au)
    mwdμeq = dμeq[1] / massWeight
    λ = sqrt(2ωc) * χ
    αi2 = (λ * mwdμeq)^2
    sumαi = αi2 * sumCosθ
    ω₊2, ω₋2, Θ = computeModesFreq(ωc, sumαi)
    # ω₊ = sqrt(ω₊2)
    # ω₋ = sqrt(ω₋2)
    λ₊ = λ * cos(Θ)
    λ₋ = -λ * sin(Θ)
    return constructForce(ω₊2, ω₋2, λ₊, λ₋, massWeight, sumCosθ)
end

function constructForce(ω₊2::T, ω₋2::T, λ₊::T, λ₋::T, mw::T, sumCosθ::T) where T<:Real
    mwλ₊ = λ₊ / mw
    mwλ₋ = λ₋ / mw
    λ₊μeq = λ₊ * μeq[1]
    λ₋μeq = λ₋ * μeq[1]
    q₊ = -λ₊μeq * sumCosθ / ω₊2
    q₋ = -λ₋μeq * sumCosθ / ω₋2
    # println(q₊, " ", q₋)
    function force3Modes!(f::T, x::T, ∑cosθ::T1) where {T<:AbstractVector{T2}, T1<:Real} where T2<:Real
        dv, μ, dμ = computeForceComponents(x[1]/mw)
        f[1] = dv / mw + (mwλ₊ * x[2] + mwλ₋ * x[3]) * dμ
        f[2] = -ω₊2 * x[2] - λ₊μeq * ∑cosθ - λ₊ * μ
        f[3] = -ω₋2 * x[3] - λ₋μeq * ∑cosθ - λ₋ * μ
    end
    return q₊, q₋, force3Modes!
end

function computeModesFreq(ωc::T, sumαi::T) where T<:Real
    wc2 = ωc^2
    w02 = ω0^2
    plus = wc2 + w02
    minus = wc2 - w02
    ac = sqrt(minus^2 + 4.0 * sumαi)
    ω₊2 = (plus + ac) / 2.0
    ω₋2 = (plus - ac) / 2.0
    Θ = atan(2.0sqrt(sumαi) / minus) / 2.0
    if Θ < 0.0
        Θ += pi / 2.0
    end
    return ω₊2, ω₋2, Θ
end

function buildBath(nBath::T1, nMolecule::T1, nb::T1, x0::AbstractArray{T2}, 
    temp::T2, dt::T2) where {T1<:Integer, T2<:Real}
    # compute coefficients for bath modes
    ω, mω2, c, c_mω2 = computeBathParameters(nBath)
    # array to store equilibrated bath coordinates for the next trajectory
    xb0 = Vector{Float64}(undef, nMolecule * nBath)
    # assgin the equilibrium positions as the initial positions of bath
    index = 0
    for i in eachindex(1:nMolecule)
        @inbounds @simd for j in eachindex(1:nBath)
            index += 1
            ix = (i-1) * nb + 1
            xb0[index] = x0[ix] * c_mω2[j]
        end
    end
    bath = Dynamics.ClassicalBathMode(nBath, amu2au, sqrt(temp/amu2au),
        ω, c, mω2, c_mω2, xb0, similar(xb0), dt/(2*amu2au), similar(xb0))
    # pre-allocated array to avoid allocations for force evaluation
    if nb == 1
        # FIXME: nMol + 1
        cache = Dynamics.SystemBathCache(Vector{Float64}(undef, nMolecule),
            similar(xb0))
    else
        cache = Dynamics.SystemBathCache(Vector{Float64}(undef, nb),
            similar(xb0))
    end
    return bath, cache
end

function getRandomAngles!(angles::AbstractVector{Float64}, rng::AbstractRNG)
    Random.rand!(rng, angles)
    # @. angles = cos(angles * 2pi)
    angles[1] = 1.0
    index = 1
    for i in 1:length(angles)÷2
        index += 1
        tmp = cos(angles[index] * 2pi) 
        angles[index] = tmp
        index += 1
        angles[index] = -tmp
    end
end

"""
    function checkParticleNumbers(nParticle::T1, nMolecule::T1, chi::T2) where {T1<:Integer, T2<:Real}

Check if the number of total particle, molecules and photons make sense. Should
not be very meaningful for now. Keep for future.
"""
function checkParticleNumbers(nParticle::T1, nMolecule::T1, chi::T2,
    model::Symbol) where {T1<:Integer, T2<:Real}

    if nParticle != nMolecule && chi != 0.0
        # χ != 0 
        if nParticle - nMolecule > 1
            println("Multi-modes not supported currently. Reduce to single-mode.")
            nMolecule = nParticle - 1
        end
        if model == :normalModes
            # 3 modes reduced Hamiltonian
            label = ["mol", "q₊", "q₋"]
            mass = [1.0, 1.0, 1.0]
            return nParticle, nMolecule, label, mass
        end
        label = ["photon"]
        mass = [1.0]
    else
        if nMolecule > 1
            println("In no-coupling case, multi-molecule does not make sense. Reduce to single-molecule.")
        end
        nParticle = 1
        nMolecule = 1
        label = Vector{String}(undef, 0)
        mass = Vector{Float64}(undef, 0)
    end
    label = vcat(repeat(["mol"], nMolecule), label)
    mass = vcat(repeat([amu2au], nMolecule), mass)
    return nParticle, nMolecule, label, mass
end

function μetp₊(x::T) where T<:Real
    return μk * (x - xc) - μ
end

function μetp₋(x::T) where T<:Real
    return μk * (x + xc) + μ
end

"""
    function computeForceComponents(x::T) where T<:Real

Compute -dvdr, μ, and -dμdr in one loop. dvdr and dμdr are negated as they
appear negative in the force expression. It is done by taking negative values
of their prefactors.
"""
function computeForceComponents(x::T, k::Integer=1) where T<:Real
    dv = 0.0
    μ = 0.0
    dμ = 0.0
    @inbounds @simd for j in eachindex(1:6)
        ϕ1 = pesCoeff[1, j, k] * x
        ϕ2 = dipoleCoeff[1, j, k] * x + dipoleCoeff[2, j, k]
        dv += pesCoeff[3, j, k] * sin(ϕ1)
        μ += dipoleCoeff[3, j, k] * sin(ϕ2)
        dμ += dipoleCoeff[4, j, k] * cos(ϕ2)
    end
    @inbounds @simd for j in 7:8
        ϕ1 = pesCoeff[1, j, k] * x
        dv += pesCoeff[3, j, k] * sin(ϕ1)
    end
    return dv, μ, dμ
end

function getPES(pes="../../pes_low.txt", dm="../../dm_low.txt")
    potentialRaw = readdlm(pes)
    dipoleRaw = readdlm(dm)
    xrange = LinRange(potentialRaw[1, 1], potentialRaw[end, 1],
        length(potentialRaw[:, 1]))
    pesMol = LinearInterpolation(xrange, potentialRaw[:, 2],
        extrapolation_bc=Line())
    xrange = LinRange(dipoleRaw[1, 1], dipoleRaw[end, 1],
        length(dipoleRaw[:, 1]))
    dipole = LinearInterpolation(xrange, dipoleRaw[:, 2],
        extrapolation_bc=Line())
    return pesMol, dipole
end

"""
    function constructForce(omegaC::T1, couple::T1, nParticle::T2, nMolecule::T2) where {T1, T2<:Real}

Function to construct the total potential of the polaritonic system for different
number of Molecules and photons.

Note that the returned potential is only for calclating force purpose, so it is
inverted to avoid a "-" at each iteration.
"""
# TODO RPMD & multi-modes?
function constructForce(nParticle::T1, nMolecule::T1, constrained::T1, ωc::T2,
    χ::T2, alignment::T3, barriers::T3) where {T1<:Integer, T2<:Float64,
    T3<:Symbol}

    if nParticle == nMolecule
        # when there is no photon, we force the system to be 1D
        # we will still use an 1-element vector as the input
        # TODO use 1 float number in this case for better performance
        """
            function forceOneD(f::AbstractVector{T}, x::AbstractVector{T},
            cache::AbstractMatrix{T}) where T<:Real

        Compute force for one single molecule and zero photon.
        Cache is dummy, since I can't get the code disptached for now.
        """
        @inline function forceOneD!(f::AbstractVector{T}, x::AbstractVector{T}
            ) where T<:Real
            f[1] = dvdr(x[1])
        end
        return forceOneD!
    else
        if nMolecule == 1
            # single-molecule and single photon
            return single(ωc, χ, false)
        else
            # multi-molecules and single photon
            return multi(nMolecule, constrained, ωc, χ, alignment, barriers)
        end
    end
end

function single(ωc::T1, χ::T1, itp::Bool) where T1<:Float64
    # compute the constants in advances. they can be expensive for many cycles
    kPho = 0.5 * ωc^2
    kPho2 = -2kPho
    # sqrt(2 / ω_c^3) * χ
    couple = sqrt(2/ωc^3) * χ
    sqrt2ωχ = -kPho2 * couple
    χ2byω = 2χ^2 / ωc
    if itp
        v, mu = getPES()
        @inline function uTotal(x::AbstractVector{T}) where T<:AbstractFloat
            return -v(x[1]) - kPho * (couple*mu(x[1]) + x[2])^2
        end
        cache = GradientCache(zeros(2), zeros(2))
        forceSingleMol! = (f, x) -> finite_difference_gradient!(f, uTotal, x, cache)
        return forceSingleMol!
    else
        """
            function forceSingleMol(f::AbstractVector{T}, x::AbstractVector{T},
            cache::AbstractMatrix{T}) where T<:Real

        Compute force for one single molecule and one photon.
        Cache is dummy, since I can't get the code disptached for now.
        """
        function forceSingleMol!(f::AbstractVector{T}, x::AbstractVector{T}
            ) where T<:Real                
            
            interaction = (couple * dipole(x[1]) + x[2])
            f[1] = dvdr(x[1]) + sqrt2ωχ * dμdr(x[1]) * interaction
            f[2] = kPho2 * interaction
        end
        return forceSingleMol!
    end
end

function multi(nMolecule::T1, constrained::T1, ωc::T2, χ::T2, alignment::T3,
    barriers::T3) where {T1<:Integer, T2<:Float64, T3<:Symbol}
    
    # compute the constants in advances. they can be expensive for many cycles
    kPho2 = -ωc^2
    sqrt2ωχ = sqrt(2ωc) * χ
    χ2byω = 2χ^2 / ωc

    if barriers == :twoBarriers
        index = repeat([3 - constrained], nMolecule)
        index[1] = constrained
    else
        index = repeat([1], nMolecule)
    end
    force = function twoBarriers!(f::T, x::T) where T<:AbstractVector{T1
        } where T1<:Real

        q = x[end]
        tmp = q * sqrt2ωχ 
        f[1] = dipole(x[1], index[1])
        f[2] = dipole(x[2], index[2])
        ∑μ = f[1] + f[2]
        @inbounds @simd for i in eachindex(1:2)
            # μ = dipole(x[i], index[i])
            # the return values of dvdr and dμdr is already negated
            dv = dvdr(x[i], index[i])
            dμ = dμdr(x[i], index[i])
            f[i] = dv + (tmp + χ2byω * ∑μ) * dμ
            # ∑μ += μ
        end
        f[end] = kPho2 * q - sqrt2ωχ * ∑μ
    end        
    """
        function disordered!(f::T, x::T, angle::T) where T<:Vector{T1
            } where T1<:Real

    The ugly but faster way of implementing disordered multi-molecule force evaluation.
    """
    if alignment == :disordered
        force = function disordered!(f::T, x::T, angle::T) where T<:Vector{T1
            } where T1<:Real

            n = length(x)-1
            ∑μ = 0.0
            q = x[end]
            tmp = q * sqrt2ωχ 
            @inbounds @simd for i in eachindex(1:n)
                ∑μ += dipole(x[i], index[i])
            end
            @inbounds @simd for i in eachindex(1:n)
                cosθ = angle[i]
                # dv, μ, dμ = computeForceComponents(x[i])
                dv = dvdr(x[i], index[i])
                dμ = dμdr(x[i], index[i])
                # f[i] = dv + tmp * dμ * cosθ
                f[i] = dv + (tmp + χ2byω * ∑μ) * dμ * cosθ
            end
            f[end] = kPho2 * q - sqrt2ωχ * ∑μ
        end            
    else
        """
            function ordered!(f::T, x::T) where T<:Vector{T1
                } where T1<:Real

        The ugly but faster way of implementing ordered multi-molecule force evaluation.
        """
        force = function ordered!(f::T, x::T) where T<:Vector{T1} where T1<:Real
            
            n = length(x)-1
            ∑μ = 0.0
            q = x[end]
            tmp = q * sqrt2ωχ 
            @inbounds @simd for i in eachindex(1:n)
                ∑μ += dipole(x[i], index[i])
            end
            @inbounds @simd for i in eachindex(1:n)
                dv = dvdr(x[i], index[i])
                dμ = dμdr(x[i], index[i])
                f[i] = dv + (tmp + χ2byω * ∑μ) * dμ
            end
            f[end] = kPho2 * q - sqrt2ωχ * ∑μ
        end
    end
    return force
end