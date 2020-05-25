using ImportMacros
using Printf
using Calculus
using Dierckx
using DelimitedFiles
using PyPlot
using LaTeXStrings
using Optim, LineSearches

struct constants
    au2wn::Float64
    au2ev::Float64
    au2k::Float64
end

mutable struct molecule
    mass::Float64
    deltaV::Float64
    omegaB::Float64
    q::Float64
    v::Float64
end

mutable struct photon
    mass::Float64
    omegaC::Float64
    chi::Float64
    q::Float64
    v::Float64
end

# TODO: should be a better way to initialize these values
consts = constants(219474.63068, 27.2113961, 3.15775e5)
mol = molecule(1836, 0.045469750862, 0.0017408, 0, 0)
pho = photon(1, 0.2/consts.au2ev, 0, 0, 0) 

# deprecated analytical solution for the imaginary ω
# μ(R) and origianl ω need to be fitted before hand
function solve(omegaC, chi)
    chi2_M = chi^2 / mol.mass
    a_bc2 = 0.67635632914003565952952644166097
    hf11 = -mol.omegaB^2 + 2chi2_M/omegaC*a_bc2
    omegaC2 = omegaC^2
    tmp1 = omegaC2 + hf11
    tmp2 = (omegaC2 - hf11)^2 + 8a_bc2*omegaC*chi2_M
    tmp2 = sqrt(tmp2)
    freq = [-sqrt((tmp2-tmp1)/2), sqrt((tmp2+tmp1)/2)] * consts.au2wn
    return freq
end

# obtain both normal mode frequencies and
# the corresponding crossover temperature
function getImFreq(x0::Array{Float64,1})
    hf = Calculus.hessian(x -> uTotal(x[1], x[2]))
    mat = hf(x0)
    mat[1, 1] /= mol.mass
    mat[2, 1] /= sqrt(mol.mass)
    mat[1, 2] = mat[2, 1]
    # possibly a bug in Calculus package
    # when ω is fairly small, ∂²V/∂q² will be ZERO
    mat[2, 2] = pho.omegaC^2
    lambda = Calculus.eigvals(mat)
    if lambda[1] < 0
        lambda[1] = -sqrt(-lambda[1])
        tC = -lambda[1] / (2pi) * consts.au2k
    else
        lambda[1] = sqrt(lambda[1])
        tC = NaN
    end
    lambda[2] = sqrt(lambda[2])
    return lambda*consts.au2wn, tC
end

function printPES()
    # dummy
end
#     qMax = 1.5x[1]
#     qMax = 3x[2]
#     println(x, ", ", χ)
#     pesFile = string("PES_", χ, ".txt")
#     mepFile = string("MEP_", χ, ".txt")
#     pesOut = open(pesFile, "w")
#     mepOut = open(mepFile, "w")
#     @printf(pesOut, "# ω_c = %5.3f\n", pho.omegaC)
#     @printf(pesOut, "# χ = %5.3f\n", pho.chi)
#     @printf(pesOut, "# minimum of PES (%5.2f, %5.2f)\n", x[1], x[2])
#     @printf(pesOut, "# imaginary frequency %5.2f cm-1, ", freq[1])
#     @printf(pesOut, "original imaginary frequency %5.2f cm-1\n", mol.omegaB*consts.au2wn)
# 
#     q = LinRange(-qMax, qMax, 101)
#     q = LinRange(-qMax, qMax, 101)
#     mep = LinearInterpolation([-qMax, qMax], [-qMax/2, qMax/2])
# 
#     for i in q
#         y = mep(i)
#         u = uTotal(i, y, halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
#         @printf(mepOut, "%5.2f %5.2f %9.6f \n", i, y, u)
#         @printf(mepOut, "\n")
#         for j in q
#             u = uTotal(i, j, halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
#             @printf(pesOut, "%5.2f %5.2f %9.6f \n", i, j, u)
#         end
#         @printf(pesOut, "\n")
#     end
# 
# end

# read discretized data
potentialRaw = readdlm("pes.txt")
dipoleRaw = readdlm("dm.txt")

# cubic spline interpolate of the PES
pesMol = Spline1D(potentialRaw[:, 1], potentialRaw[:, 2])
# cubic spline interpolate of μ
dipole = Spline1D(dipoleRaw[:, 1], dipoleRaw[:, 2])

# photon DOF potential
pesPho(qPho) = 0.5 * pho.omegaC^2 * qPho^2
# light-mater interaction
lightMatter(qMol, qPho) = sqrt(2*pho.omegaC)*pho.chi*dipole(qMol)*qPho +
    pho.chi^2*dipole(qMol)^2/pho.omegaC
# total potential
uTotal(qMol, qPho) = pesMol(qMol) + pesPho(qPho) + lightMatter(qMol, qPho)

# verify we can get correct ω
# TODO: could be moved to unit tests later
# pho.chi = 0.002
# pho.omegaC = 5e-5
# println(getImFreq())
# println(solve(pho.omegaC, pho.chi))
# println(sqrt(-second_derivative(pesMol, 0.0)/mol.mass)*consts.au2wn)
chi = append!(collect(0.0000:0.0002:0.0008), collect(0.001:0.001:0.01))
omega = collect(0.2:0.1:6) * mol.omegaB
 
# calculate the relation between χ, ω_c and ω_b 
function omegaOmegaC(chi, omegaC)
    nChi = length(chi)
    nOmega = length(omegaC)
    imFreq = Array{Float64, 2}(undef, nChi, nOmega)
    f = open("omegac", "w")
    for j in 1:nOmega
        @printf(f, "%8.3f", omegaC[j]*consts.au2wn)
        for i in 1:nChi
            pho.chi = chi[i]
            pho.omegaC = omegaC[j]
            tmp, tC = getImFreq()
            imFreq[i, j] = tmp[1]
            @printf(f, " %8.3f", tmp[1])
        end
        @printf(f, "\n")
    end
    for k in 1:length(chi)
        plot(omega*consts.au2wn, imFreq[k, :], label=string(chi[k]))
    end
    grid(true)
    xlabel(L"$\omega_c$")
    ylabel(L"$\omega_b$")
    legend()
    # avoid conflict with Base.show()
    PyPlot.show()
end

"""
# optimize PES
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

"""
# find the transition state on the PES
"""
# FIXME: 1. optimizer boundaries
#        2. Lower and upper bounds in interpolation
function optPES(targetFunc::Function, x0::Array{Float64, 1},
    x1::Array{Float64, 1}, algor; maxTries=5)
    local minLoc, minVal, freq, temp

    n = 0
    xBounds = sort([x0[1], x1[1]])
    yBounds = sort([x0[2], x1[2]])
    xi = Vector{Float64}(undef, 2)
    itp = Spline1D(xBounds, yBounds, k=1)
    x0[2] -= 0.5
    x1[2] += 0.5
    while n < maxTries
        xi[1] = rand() - 0.5
        xi[2] = itp(xi[1])
        println(xi)
        result = optimize(x -> -targetFunc(x[1], x[2]), x0, x1, xi,
            Fminbox(algor()))
        minLoc = Optim.minimizer(result)
        minVal = Optim.minimum(result)
        if Optim.converged(result)
            freq, temp = getImFreq(minLoc)
            if freq[1] * freq[2] < 0
                break
            end
        else
            x0 = minLoc
            n += 1
            if n > maxTries
                throw("Hit max number of tries in `optPES`.
                    Consider using different parameters.")
            end
        end
    end
    return minLoc, minVal, freq, temp
end

# omegaOmegaC(chi, omega)
pho.chi = 0.0
function extrema(targetFunc, method)
    # TODO: add try catch stuff
    # TODO: throw this to a function like optPES
    maxTries = 5
    # find the minimum in the first quadrant
    x0 = rand(Float64, 2)
end

# result = optimize(x -> -uTotal(x[1], x[2]), [-1.73, -Inf], [1.73, Inf], x0, Fminbox(LBFGS()))
# println(Optim.minimizer(result), Optim.converged(result), Optim.iterations(result))

minLoc, minVal = optPES(uTotal, [1.0, 0], BFGS)
out = optPES(uTotal, -minLoc, minLoc, LBFGS)
println(out)