using ImportMacros
using Printf
using Calculus
using Dierckx
using DelimitedFiles
using PyPlot
using LaTeXStrings
@import LinearAlgebra as la

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

consts = constants(219474.63068, 27.2113961, 3.15775e5)
mol = molecule(1836, 0.045469750862, 0.0017408, 0, 0)
pho = photon(1, 0.2/consts.au2ev, 0, 0, 0) 

function solve(omegaC, chi)
    chi2_M = chi^2 / mol.mass
    a_bc2 = 0.67635632914003565952952644166097
    hf11 = -mol.omegaB^2 + 2chi2_M/omegaC*a_bc2
    omegaC2 = omegaC^2
    tmp1 = omegaC2 + hf11
    # println(hf11, " ", sqrt(2omegaC*a_bc2/mol.mass)*chi)
    tmp2 = (omegaC2 - hf11)^2 + 8a_bc2*omegaC*chi2_M
    tmp2 = sqrt(tmp2)
    #println((tmp2-tmp1)/2)
    freq = [-sqrt((tmp2-tmp1)/2), sqrt((tmp2+tmp1)/2)] * consts.au2wn
    return freq
end

function getImFreq()
    hf = Calculus.hessian(x -> uTotal(x[1], x[2]))
    mat = hf([0.0, 0.0])
    mat[1, 1] /= mol.mass
    mat[2, 1] /= sqrt(mol.mass)
    mat[1, 2] = mat[2, 1]
    mat[2, 2] = pho.omegaC^2
    lambda = Calculus.eigvals(mat)
    #println(lambda, " lambda")
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

function stpdest(a, b, c)
    tol = 1e-8
    maxCycle = 100000
    i = 0
    step = 5
    x = [0.1, 0.2]
    g = Calculus.gradient(x -> uTotal(x[1], x[2]), x)
    gNorm = la.norm(g)
    while (gNorm > tol)
        i += 1
        x -= g * step
        g = Calculus.gradient(x -> uTotal(x[1], x[2]), x)
        gNorm = la.norm(g)
        if (i > maxCycle)
            println("Fail to converge after $maxCycle iterations.")
            return
        end
    end
    println("Converge after $i iterations.")
    return x
end

# getImFreq(halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
# freq = solve(pho.omegaC, pho.chi)
# println(freq, " freq anal")
# chi = collect(0.000:0.002:0.008)
# for χ in chi
#     pho.chi = χ
#     halfOmegaC2 = pho.omegaC^2 / 2
#     sqrt2OmegaChi = sqrt(2pho.omegaC) * pho.chi
#     chi2OverOmega  = pho.chi^2 / pho.omegaC
#     x = stpdest(halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
#     freq = solve(pho.omegaC, pho.chi)
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
# pesMol(x) = 4.254987695360661e-5x^4 - 0.00278189309952x^2
# dipole(x) = 1.9657x - 25.2139tanh(x/9.04337)

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
    return imFreq
end

imFreq = omegaOmegaC(chi, omega)
for k in 1:length(chi)
    plot(omega*consts.au2wn, imFreq[k, :], label=string(chi[k]))
end
grid(true)
xlabel(L"$\omega_c$")
ylabel(L"$\omega_b$")
legend()
show()
