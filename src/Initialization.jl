using Constants
using Random
using LinearAlgebra: dot

# Fourier series fitted parameters
# dipole
const dipoleCoeff = [0.8017194824351892 3.2923071635398693 1.9571015898154989 4.721532196133278 7.745224719320653 6.217082808943054;
    3.1415926535894014 3.1415926535945484 -3.1415926535898957 3.141592653598072 3.141592653535209 -1.760747833413431e-11;
    1.3496422641395616 0.07841064324878252 0.31897867072301 0.019819260283106183 0.0012061473577049471 -0.005001264129667085;
    -1.0820344974786262 -0.2581519224657358 -0.6242736635892374 -0.09357727553023139 -0.009341882330039646 0.031093273243536782]
# potenial
const a0 = 9.92928333994269962659 
const pesCoeff = [0.4480425396401699 0.8960850792803398 1.3441276189205096 1.7921701585606795 2.2402126982008492 2.688255237841019 3.136297777481189 3.584340317121359;
    -19.07993156151155 14.132538112430545 -8.669598196769785 4.440544955027029 -1.841789733545533 0.5938508511955884 -0.13456854929031054 0.017221210502076565;
    -8.548620992980267 12.663956534909747 -11.653046381221715 7.958212156146616 -4.126000748504662 1.5964226612228885 -0.42204704205806876 0.0617266791122268]

# equilibirium position of R coordinate under the current potenial
const xeq = -1.735918600503033
const ω0 = 0.0062736666471555754
const μeq = 1.2197912355997298
const dμ0 = -2.0984725146374075
const dμeq = 0.2253318892690798
const freqCutoff = 500.0 / au2wn
const eta = 4.0 * amu2au * freqCutoff
const gamma = eta / amu2au / 5.0

"""
    function initialize(nParticle::T1, temp::T2, ωc::T2, chi::T2) where {T1<:Integer, T2<:Real}

Initialize most of the values, parameters and structs for the dynamics.
"""
# TODO better and more flexible way to handle input values
function initialize(nParticle::T1, temp::T2, ωc::T2, chi::T2;
    dynamics::Symbol=:langevin, model::Symbol=:normalModes,
    alignment::Symbol=:ordered) where {T1<:Integer, T2<:Real}

    nPhoton = 1
    nMolecule = nParticle - nPhoton
    # number of bath modes per molecule! total number of bath mdoes = nMolecule * nBath
    nBath = 15
    dt = 4.0
    ν = 0.05
    τ = floor(Int64, 1/ν)
    # collisionFrequency 
    z = ν * dt

    # check if the number of molecules and photons make sense
    nParticle, nMolecule, label, mass = checkParticleNumbers(nParticle,
        nMolecule, chi, model)
    # total number of bath modes
    nBathTotal = nMolecule * nBath

    # the suffix for all output file names to identify the setup and avoid overwritting
    flnmID = string(ωc, "_", chi, "_", temp, "_", nMolecule)
    # convert values to au, so we can keep more human-friendly values outside
    ωc /= au2ev
    chi /= au2ev
    temp /= au2kelvin
    couple = sqrt(2/ωc^3) * chi
    # println("+: ", λ₊^2 * nMolecule * μeq * dμ0 / sqrt(amu2au) / ω₊, " ", ω₊)
    # println("-: ", λ₋^2 * nMolecule * μeq * dμ0 / sqrt(amu2au) / ω₋, " ", ω₋)

    rng = Random.seed!(1233+Threads.threadid())
    if alignment == :ordered
        angles = ones(nMolecule)
        sumCosθ = convert(Float64, nMolecule)
    else
        angles = Vector{Float64}(undef, nMolecule)
        getRandomAngles!(angles, rng)
        sumCosθ = sum(angles)
    end
    # array to store equilibrated molecule and photon coordinates for the next trajectory
    x0 = repeat([xeq], nParticle)
    x0[1] = 0.0
    if nMolecule > 1
        x0[end] = couple * -μeq * (sumCosθ-1)
    else
        x0[end] = 0.0
    end

    if dynamics == :SystemBath
        bath, cache = buildBath(nBath, nMolecule, x0, temp, dt)
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
        nBathTotal, 1, 1, 1)
    mol = Dynamics.FullSystemParticle(nMolecule, label, mass, sqrt.(temp./mass),
        x0, similar(mass), param.Δt./(2*mass), similar(mass), angles)
    # obatin the gradient of the corresponding potential
    forceEvaluation! = constructForce(ωc, chi, nParticle, nMolecule)
    return rng, param, mol, bath, forceEvaluation!, cache, flnmID
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
    tmp = sqrt(2eta * amu2au * freqCutoff / nb / pi)
    @inbounds @simd for i in eachindex(1:nBath)
        # bath mode frequencies
        ω[i] = -freqCutoff * log((i-0.5) / nb)
        # bath force constants
        mω2[i] = ω[i]^2 * amu2au
        # bath coupling strength
        c[i] = tmp * ω[i]
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
    mwdμeq = dμeq / massWeight
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
    λ₊μeq = λ₊ * μeq
    λ₋μeq = λ₋ * μeq
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

function buildBath(nBath::T1, nMolecule::T1, x0::AbstractVector{T2}, temp::T2,
    dt::T2) where {T1<:Integer, T2<:Real}
    # compute coefficients for bath modes
    ω, mω2, c, c_mω2 = computeBathParameters(nBath)
    # array to store equilibrated bath coordinates for the next trajectory
    xb0 = Vector{Float64}(undef, nMolecule * nBath)
    # assgin the equilibrium positions as the initial positions of bath
    index = 0
    for i in eachindex(1:nMolecule)
        @inbounds @simd for j in eachindex(1:nBath)
            index += 1
            xb0[index] = x0[i] * c_mω2[j]
        end
    end
    bath = Dynamics.ClassicalBathMode(nBath, amu2au, sqrt(temp/amu2au),
        ω, c, mω2, c_mω2, xb0, similar(xb0), dt/(2*amu2au), similar(xb0))
    # pre-allocated array to avoid allocations for force evaluation
    cache = Dynamics.SystemBathCache(Vector{Float64}(undef, nMolecule),
        similar(xb0))
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
            nParticle = 1
            nMolecule = 1
        end
        label = Vector{String}(undef, 0)
        mass = Vector{Float64}(undef, 0)
    end
    label = vcat(repeat(["mol"], nMolecule), label)
    mass = vcat(repeat([amu2au], nMolecule), mass)
    return nParticle, nMolecule, label, mass
end

"""
    function dipole(x::T) where T<:Real

Fourier sine series to compute the permannet dipole at x.
"""
function dipole(x::T) where T<:Real
    mu = 0.0
    @inbounds @simd for i in 1:6
        ϕ = dipoleCoeff[1, i] * x + dipoleCoeff[2, i]
        mu += dipoleCoeff[3, i] * sin(ϕ)
    end
    return mu
end

"""
    function pes(x::T) where T<:Real

Fourier cosine series to compute the PES at x.
"""
function pes(x::T) where T<:Real
    v = a0
    @inbounds @simd for i in eachindex(1:8)
        ϕ = pesCoeff[1, i] * x
        v += pesCoeff[2, i] * cos(ϕ)
    end
    return v
end

function ufit(x::AbstractVector{T}) where {T<:Real}
    ∑μ = 0.0
    ∑v = 0.0
    @inbounds @simd for i in eachindex(1:length(x)-1)
        ∑μ += dipole(x[i])
        ∑v += pes(x[i])
    end
    return ∑v + 1.728705305973675e-5 * (15.055424826054347 * ∑μ + x[end])^2
end
"""
    function dvdr(x::T) where T<:Real

Compute the derivative with respect to PES at x.
"""
function dvdr(x::T) where T<:Real
    dv = 0.0
    @inbounds @simd for i in eachindex(1:8)
        ϕ = pesCoeff[1, i] * x
        dv += pesCoeff[3, i] * sin(ϕ)
    end
    return dv
end

"""
    function dμdr(x::T) where T<:Real

Compute the derivative with respect to permannet dipole at x.
"""
function dμdr(x::T) where T<:Real
    dμ = 0.0
    for i in eachindex(1:6)
        ϕ = dipoleCoeff[1, i] * x + dipoleCoeff[2, i]
        dμ += dipoleCoeff[4, i] * cos(ϕ)
    end
    return dμ
end

"""
    function computeForceComponents(x::T) where T<:Real

Compute -dvdr, μ, and -dμdr in one loop. dvdr and dμdr are negated as they
appear negative in the force expression. It is done by taking negative values
of their prefactors.
"""
function computeForceComponents(x::T) where T<:Real
    dv = 0.0
    μ = 0.0
    dμ = 0.0
    @inbounds @simd for j in eachindex(1:6)
        ϕ1 = pesCoeff[1, j] * x
        ϕ2 = dipoleCoeff[1, j] * x + dipoleCoeff[2, j]
        dv += pesCoeff[3, j] * sin(ϕ1)
        μ += dipoleCoeff[3, j] * sin(ϕ2)
        dμ += dipoleCoeff[4, j] * cos(ϕ2)
    end
    @inbounds @simd for j in 7:8
        ϕ1 = pesCoeff[1, j] * x
        dv += pesCoeff[3, j] * sin(ϕ1)
    end
    return dv, μ, dμ
end

"""
    function constructForce(omegaC::T1, couple::T1, nParticle::T2, nMolecule::T2) where {T1, T2<:Real}

Function to construct the total potential of the polaritonic system for different
number of Molecules and photons.

Note that the returned potential is only for calclating force purpose, so it is
inverted to avoid a "-" at each iteration.
"""
# TODO RPMD & multi-modes?
function constructForce(ωc::T1, χ::T1, nParticle::T2, nMolecule::T2) where {T1, T2<:Real}
    # compute the constants in advances. they can be expensive for many cycles
    kPho = 0.5 * ωc^2
    # sqrt(2 / ω_c^3) * χ
    kPho2 = -2kPho
    couple = sqrt(2/ωc^3) * χ
    sqrt2ωχ = -kPho2 * couple
    χ2byω = 2χ^2 / ωc
    # construct an inverted total potential
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

    function pot(x::AbstractVector{T}) where {T<:Real}
        return pes(x[1]) + kPho * (couple * dipole(x[1]) + x[2])^2
    end

    function energy(x::AbstractVector{T}, v::AbstractVector{T}) where {T<:Real}
        return 918.0v[1]^2 + 0.5v[2]^2 + pot(x)
    end

    """
        function forceiMultiMol(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real

    The ugly but faster way of implementing multi-molecule force evaluation.
    """
    function forceMultiMol!(f::T, x::T, angle::T) where T<:AbstractVector{T1
        } where T1<:Real

        ∑μ = 0.0
        q = x[end]
        tmp = q * sqrt2ωχ 
        @inbounds @simd for i in eachindex(1:length(x)-1)
            cosθ = angle[i]
            dv, μ, dμ = computeForceComponents(x[i])
            # f[i] = dv + (tmp + χ2byω * μ) * dμ
            f[i] = dv + tmp * dμ * cosθ
            ∑μ += μ * cosθ
        end
        f[end] = kPho2 * q - sqrt2ωχ * ∑μ
    end

    function forceMultiMol!(f::T, x::T) where T<:AbstractVector{T1} where T1<:Real

        ∑μ = 0.0
        q = x[end]
        tmp = q * sqrt2ωχ
        @inbounds @simd for i in eachindex(1:length(x)-1)
            dv, μ, dμ = computeForceComponents(x[i])
            # f[i] = dv + (tmp + χ2byω * μ) * dμ
            f[i] = dv + tmp * dμ
            ∑μ += μ 
        end
        f[end] = kPho2 * q - sqrt2ωχ * ∑μ
    end
    if nParticle == nMolecule
        # when there is no photon, we force the system to be 1D
        # we will still use an 1-element vector as the input
        return forceOneD!
    else
        if nMolecule == 1
            # single-molecule and single photon
            return forceSingleMol!
        else
            # multi-molecules and single photon
            return forceMultiMol!
        end
    end
end