using Printf
using Calculus
using ImportMacros
@import LinearAlgebra as LA
using Dierckx: Spline1D
using DelimitedFiles
using PyPlot
# using IntervalArithmetic, IntervalRootFinding
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
mol = molecule(1836, 0.0, 1307.404/consts.au2wn, 0, 0)
pho = photon(1, 0.04/consts.au2ev, 0, 0, 0) 

# deprecated analytical solution for the imaginary ω
# μ(R) and origianl ω need to be fitted before hand
function solve(omegaC, chi)
    chi2_M = chi^2 / mol.mass
    a_bc2 = 2.1027274887558707^2
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
    # possibly a bug in Calculus package
    # when ω is fairly small, ∂²V/∂q² will be ZERO
    mat[2, 2] = pho.omegaC^2
    mat_og = deepcopy(mat)
    mat[1, 1] /= mol.mass
    mat[2, 1] /= sqrt(mol.mass)
    mat[1, 2] = mat[2, 1]
    lambda = Calculus.eigvals(mat)
    v = Calculus.eigvecs(mat)
    if lambda[1] < 0
        lambda[1] = -sqrt(-lambda[1])
        tC = -lambda[1] / (2pi) * consts.au2k
    else
        lambda[1] = sqrt(lambda[1])
        tC = NaN
    end
    lambda[2] = sqrt(lambda[2])
    return lambda*consts.au2wn, v, tC
end

# FIXME: MEP implementaion is completely wrong
function printPES(xMin1, xMin2)
    pesFile = string("PES_", pho.chi, ".txt")
    # mepFile = string("MEP_", pho.chi, ".txt")
    pesOut = open(pesFile, "w")
    # mepOut = open(mepFile, "w")
    @printf(pesOut, "# ω_c = %5.3f\n", pho.omegaC)
    @printf(pesOut, "# χ = %5.3f\n", pho.chi)
    @printf(pesOut, "# minimum of PES (%5.2f, %5.2f)\n", xMin1[1], xMin1[2])
    # @printf(pesOut, "# imaginary frequency %5.2f cm-1, ", freq[1])
    # @printf(pesOut, "original imaginary frequency %5.2f cm-1\n", mol.omegaB*consts.au2wn)
    qMolMax = 5.0 # 1.5xMin1[1]
    qPhoMax = 1500.0 # 4xMin1[2]
    qMol = LinRange(-qMolMax, qMolMax, 201)
    qPho = LinRange(-qPhoMax, qPhoMax, 1001)
    # mep = Spline1D([-qMolMax, qMolMax], [-qPhoMax/2, qPhoMax/2], k=1)
    for i in qMol
        # y = mep(i)
        # u = uTotal(i, y)
        # @printf(mepOut, "%5.2f %5.2f %9.6f \n", i, y, u)
        # @printf(mepOut, "\n")
        for j in qPho
            u = uTotal(i, j)
            @printf(pesOut, "%5.2f %5.2f %9.6f \n", i, j, u)
        end
        @printf(pesOut, "\n")
    end
    # close(mepOut) 
    close(pesOut) 
end
 
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
            # minLoc1, minVal = optPES(uTotal, [1.0, 1.0], BFGS)
            # minLoc2, minVal = optPES(uTotal, [-1.0, -1.0], BFGS)
            # ts, dv, tmp, temp = optPES(uTotal, minLoc1, minLoc2, LBFGS)
            tmp, tC = getImFreq([0.0, 0.0])
            # tmp = solve(pho.omegaC, pho.chi)
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
# FIXME:
function optPES(targetFunc::Function, x0::Array{Float64, 1},
    x1::Array{Float64, 1}, algor; maxTries=5, maxCycles=100, tol=1e-12)
    local minLoc, minVal, freq, temp

    # println(x0, x1)
    if abs(x0[1] - x1[1]) < eps()
        throw("Need two different local mimima to find the TS.")
    end
    xi, xBounds, yBounds = [Vector{Float64}(undef, 2) for i = 1:3]
    # tmp = hcat(x0, x1)
    # lower = vec(minimum(tmp, dims=2))
    # upper = vec(maximum(tmp, dims=2))
    if x0[1] < x1[1]
        xBounds = [x0[1], x1[1]]
        yBounds = [x0[2], x1[2]]
    else
        xBounds = [x1[1], x0[1]]
        yBounds = [x1[2], x0[2]]
    end
    itp = Spline1D(xBounds, yBounds, k=1)
    # if abs(x0[2] - x1[2]) < eps()
    #     lower[2] -= 10eps()
    #     upper[2] += 10eps()
    # end
    hf = Calculus.hessian(x -> targetFunc(x[1], x[2]))
    grad = Calculus.gradient(x -> targetFunc(x[1], x[2]))
    n = 0
    local  i
    while n < maxTries
        i = 0
        xi[1] = rand() - 0.5
        xi[2] = itp(xi[1])
        g = grad(xi)
        while i < maxCycles
            h = hf(xi)
            dx = h \ g
            xi -= dx
            g = grad(xi)
            if LA.norm(g) < tol
                freq, temp = getImFreq(xi)
                if prod(freq) < 0
                    return xi, uTotal(xi[1], xi[2]), freq, temp
                else
                    break
                end
            end
            i += 1
            println(i, " ", xi)
        end
        n += 1
    end
    throw("Hit max number of tries in `optPES`. Consider using different parameters.")
        # result = optimize(x -> -targetFunc(x[1], itp(x[1])), lower, upper, xi,
        #     Fminbox(algor()))
        # minLoc = Optim.minimizer(result)
        # minVal = Optim.minimum(result)
        # println(result)
        # if Optim.converged(result)
        #     freq, temp = getImFreq(minLoc)
        #     if freq[1] * freq[2] < 0
        #         break
        #     end
        # else
        #     x0 = minLoc
        #     n += 1
        #     if n > maxTries
        #         throw("Hit max number of tries in `optPES`.
        #             Consider using different parameters.")
        #     end
        # end
end

# omegaOmegaC(chi, omega)
function extrema(targetFunc, method)
    # TODO: add try catch stuff
    # TODO: throw this to a function like optPES
    maxTries = 5
    # find the minimum in the first quadrant
    x0 = rand(Float64, 2)
end

# result = optimize(x -> -uTotal(x[1], x[2]), [-1.5, 0], [1.5, 0], [1.0, 0.0], Fminbox(LBFGS()))
# println(Optim.minimizer(result), Optim.converged(result), Optim.iterations(result))

# read discretized data
cd("..")
potentialRaw = readdlm("pes.txt")
dipoleRaw = readdlm("dm.txt")

# cubic spline interpolate of the PES
pesMol = Spline1D(potentialRaw[:, 1], potentialRaw[:, 2])
# cubic spline interpolate of μ
dipole = Spline1D(dipoleRaw[:, 1], dipoleRaw[:, 2])

# photon DOF potential
pesPho(qPho) = 0.5 * pho.omegaC^2 * qPho^2
# light-mater interaction
lightMatter((qMol, qPho)) = sqrt(2*pho.omegaC)*pho.chi*dipole(qMol)*qPho +
    pho.chi^2*dipole(qMol)^2/pho.omegaC
lightMatter(qMol, qPho) = sqrt(2*pho.omegaC)*pho.chi*dipole(qMol)*qPho +
    pho.chi^2*dipole(qMol)^2/pho.omegaC
# total potential
uTotal((qMol, qPho)) = pesMol(qMol) + pesPho(qPho) + lightMatter((qMol, qPho))
uTotal(qMol, qPho) = pesMol(qMol) + pesPho(qPho) + lightMatter(qMol, qPho)

pho.chi = 0.02
minLoc1, minVal = optPES(uTotal, [1.0, 1.0], BFGS)
# minLoc2, minVal = optPES(uTotal, [-1.0, -1.0], BFGS)
# out = optPES(uTotal, minLoc1, minLoc2, LBFGS)
# println(minLoc1, minLoc2, out)
# println(out)

# verify we can get correct ω
# TODO: could be moved to unit tests later
# pho.omegaC = 500000
# println(getImFreq([0.0, 0.0]))
# println(solve(pho.omegaC, pho.chi))
printPES(minLoc1, minLoc1)
# hf = Calculus.gradient(x -> uTotal(x[1], x[2]))
# ∇u = ∇(uTotal)
# println(roots(hf, IntervalBox(-20..20, 2), Newton, 1e-5))
# println(solve(pho.omegaC, pho.chi))
# println(sqrt(-second_derivative(pesMol, 0.0)/mol.mass)*consts.au2wn)
# chi = append!(collect(0.0000:0.0002:0.0008), collect(0.001:0.001:0.01))
# omega = collect(0.2:0.1:6) * mol.omegaB
# println(derivative(dipole, 0.0))
# omegaOmegaC(chi, omega)
