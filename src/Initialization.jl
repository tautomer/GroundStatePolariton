using Interpolations: LinearInterpolation, Line, AbstractExtrapolation
using Constants

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
const mueq = -1.2197912355997298
const freqCutoff = 500.0 / au2wn
const eta = 4.0 * amu2au * freqCutoff

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
    function initialize(nParticle::T1, temp::T2, ωc::T2, chi::T2) where {T1<:Integer, T2<:Real}

Initialize most of the values, parameters and structs for the dynamics.
"""
# TODO better and more flexible way to handle input values
function initialize(nParticle::T1, temp::T2, ωc::T2, chi::T2) where {T1<:Integer, T2<:Real}

    nPhoton = 1
    nMolecule = nParticle - nPhoton
    # number of bath modes per molecule! total number of bath mdoes = nMolecule * nBath
    nBath = 15
    dt = 4.0

    # compute coefficients for bath modes
    ω, mω2, c, c_mω2 = computeBathParameters(nBath)

    # check if the number of molecules and photons make sense
    nParticle, nMolecule, label, mass = checkParticleNumbers(nParticle, nMolecule, chi)

    # the suffix for all output file names to identify the setup and avoid overwritting
    flnmID = string(ωc, "_", chi, "_", temp, "_", nMolecule, ".txt")
    # convert values to au, so we can keep more human-friendly values outside
    ωc /= au2ev
    temp /= au2kelvin
    couple = sqrt(2/ωc^3) * chi

    # array to store equilibrated molecule and photon coordinates for the next trajectory
    x0 = repeat([xeq], nParticle)
    x0[1] = 0.0
    if nMolecule > 1
        x0[end] = couple * mueq * (nMolecule-1)
    else
        x0[end] = 0.0
    end
    # total number of bath modes
    nBathTotal = nMolecule * nBath
    # array to store equilibrated bath coordinates for the next trajectory
    xb0 = Vector{Float64}(undef, nBathTotal)
    index = 0
    for i in eachindex(1:nMolecule)
        @inbounds @simd for j in eachindex(1:nBath)
            index += 1
            xb0[index] = x0[i] * c_mω2[j]
        end
    end

    param = Dynamics.Parameters(temp, dt, nParticle, nMolecule, nBathTotal, 1, 1, 1)
    bath = Dynamics.ClassicalBathMode(nBath, amu2au, sqrt(temp/amu2au),
        ω, c, mω2, c_mω2, xb0, similar(xb0), param.Δt/(2*amu2au),
        similar(xb0))
    mol = Dynamics.ClassicalParticle(nMolecule, label, mass, sqrt.(temp./mass),
        x0, similar(mass), param.Δt./(2*mass), similar(mass))
    # obatin the gradient of the corresponding potential
    forceEvaluation! = constructForce(ωc, couple, nParticle, nMolecule)
    # pre-allocated array to avoid allocations for force evaluation
    cache = Matrix{Float64}(undef, 2, nMolecule)
    return param, mol, bath, forceEvaluation!, cache, flnmID
end

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

"""
    function checkParticleNumbers(nParticle::T1, nMolecule::T1, chi::T2) where {T1<:Integer, T2<:Real}

Check if the number of total particle, molecules and photons make sense. Should
not be very meaningful for now. Keep for future.
"""
function checkParticleNumbers(nParticle::T1, nMolecule::T1, chi::T2) where {T1<:Integer, T2<:Real}
    if nParticle != nMolecule && chi != 0.0
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
    return nParticle, nMolecule, label, mass
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

using BenchmarkTools
"""
    function constructForce(omegaC::T1, couple::T1, nParticle::T2, nMolecule::T2) where {T1, T2<:Real}

Function to construct the total potential of the polaritonic system for different
number of Molecules and photons.

Note that the returned potential is only for calclating force purpose, so it is
inverted to avoid a "-" at each iteration.
"""
# TODO RPMD & multi-modes?
function constructForce(omegaC::T1, couple::T1, nParticle::T2, nMolecule::T2) where {T1, T2<:Real}
    # compute the constants in advances. they can be expensive for many cycles
    kPho = 0.5 * omegaC^2
    # sqrt(2 / ω_c^3) * χ
    kPho2 = -2kPho
    sqrt2wcchi = -kPho2 * couple
    # construct an inverted total potential
    # TODO use 1 float number in this case for better performance
    """
        function forceOneD(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real

    Compute force for one single molecule and zero photon.
    Cache is dummy, since I can't get the code disptached for now.
    """
    @inline function forceOneD!(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real
        f[1] = dvdr(x[1])
    end

    """
        function forceSingleMol(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real

    Compute force for one single molecule and one photon.
    Cache is dummy, since I can't get the code disptached for now.
    """
    @inline function forceSingleMol!(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real

        interaction = (couple * dipole(x[1]) + x[2])
        f[1] = dvdr(x[1]) + sqrt2wcchi * dμdr(x[1]) * interaction
        f[2] = kPho2 * interaction
    end

    """
        function forceiMultiMol(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real

    The ugly but faster way of implementing multi-molecule force evaluation.
    A tidier way of coding is like below
    function ∇u!(f::AbstractVector{T}, x::AbstractVector{T}) where {T<:Real}
        ∑μ = 0.0
        @inbounds @simd for i in eachindex(1:length(x)-1)
            ∑μ += dipole2(x[i])
        end
        interaction = (couple * ∑μ + x[end])
        @inbounds @simd for i in eachindex(1:length(x)-1)
            f[i] = dvdr(x[i]) + sqrt2wcchi * dμdr(x[i]) * interaction
        end
        f[end] = kPho2 * interaction
    end
    For 100-mol, the ugly way is 1 μs (~10%) faster
    """
    function forceMultiMol!(f::AbstractVector{T}, x::AbstractVector{T},
        cache::AbstractMatrix{T}) where T<:Real

        ∑μ = 0.0
        @inbounds @simd for i in eachindex(1:length(x)-1)
            xi = x[i]
            dv = 0.0
            μ = 0.0
            dμ = 0.0
            @inbounds @simd for j in eachindex(1:6)
                ϕ1 = pesCoeff[1, j] * xi
                ϕ2 = dipoleCoeff[1, j] * xi + dipoleCoeff[2, j]
                dv += pesCoeff[3, j] * sin(ϕ1)
                μ += dipoleCoeff[3, j] * sin(ϕ2)
                dμ += dipoleCoeff[4, j] * cos(ϕ2)
            end
            @inbounds @simd for j in 7:8
                ϕ1 = pesCoeff[1, j] * xi
                dv += pesCoeff[3, j] * sin(ϕ1)
            end
            cache[1, i] = dv
            cache[2, i] = dμ # * (-1)^i
            ∑μ += μ # * (-1)^i
        end
        interaction = couple * ∑μ + x[end]
        tmp = interaction * sqrt2wcchi
        @inbounds @simd for i in eachindex(1:length(x)-1)
            f[i] = cache[1, i] + tmp * cache[2, i]
        end
        f[end] = kPho2 * interaction
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