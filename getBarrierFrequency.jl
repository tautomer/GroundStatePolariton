using ImportMacros
using Printf
using Calculus
using Interpolations
@import LinearAlgebra as LA

struct constants
    au2wn::Float64
    au2ev::Float64
end

mutable struct molecule
    mass::Float64
    deltaV::Float64
    omegaB::Float64
    qMol::Float64
    vMol::Float64
end

struct potential
    a::Float64
    b::Float64
end

struct dipole
    a::Float64
    b::Float64
    c::Float64
end

mutable struct photon
    mass::Float64
    omegaC::Float64
    chi::Float64
    qPho::Float64
    vPho::Float64
end

consts = constants(219474.63068, 27.2113961)
mol = molecule(1836, 0.045469750862, 0.0017408, 0, 0)
dm = dipole(-1.9657, -25.2139, 9.04337)
pot = potential(mol.mass^2*mol.omegaB^4/(16mol.deltaV), -mol.mass*mol.omegaB^2/2)
pho = photon(1, 0.2/consts.au2ev, 0, 0, 0) 

function solve(omegaC, chi)
    chi2_M = chi^2 / mol.mass
     a_bc2 = (dm.a - dm.b/dm.c)^2
    hf11 = -mol.omegaB^2 + 2chi2_M/omegaC*a_bc2
    omegaC2 = omegaC^2
    tmp1 = omegaC2 + hf11
    tmp2 = (omegaC2 - hf11)^2 + 8a_bc2*omegaC*chi2_M
    tmp2 = sqrt(tmp2)
    freq = [-sqrt((tmp2-tmp1)/2), sqrt((tmp2+tmp1)/2)] * consts.au2wn
    return freq
end

function getImFreq(halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
    hf = Calculus.hessian(x -> uTotal(x[1], x[2], halfOmegaC2, sqrt2OmegaChi, chi2OverOmega))
    mat = hf([0.0, 0.0])
    mat[1, 1] /= mol.mass
    mat[2, 1] /= sqrt(mol.mass)
    mat[1, 2] = mat[2, 1]
    println(mat[1, 1], " ", mat[2, 1])
    lambda = Calculus.eigvals(mat)
    lambda = [-sqrt(-lambda[1]), sqrt(lambda[2])]
    println(lambda*consts.au2wn, "freq numer")
end

function uTotal(qMol, qPho, halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
    x2 = qMol^2
    vMol = pot.a*x2^2 + pot.b*x2
    vPho = halfOmegaC2 * qPho^2
    mu = dm.a*qMol - dm.b*tanh(qMol/dm.c)
    inter = sqrt2OmegaChi*mu*qPho + chi2OverOmega*mu^2
    return vMol + vPho + inter
end

function stpdest(a, b, c)
    tol = 1e-8
    maxCycle = 100000
    i = 0
    step = 5
    x = [0.1, 0.2]
    g = Calculus.gradient(x -> uTotal(x[1], x[2], a, b, c), x)
    gNorm = LA.norm(g)
    while (gNorm > tol)
        i += 1
        x -= g * step
        g = Calculus.gradient(x -> uTotal(x[1], x[2], a, b, c), x)
        gNorm = LA.norm(g)
        if (i > maxCycle)
            println("Fail to converge after $maxCycle iterations.")
            return
        end
    end
    println("Converge after $i iterations.")
    return x
end

pho.chi = 0.002
halfOmegaC2 = pho.omegaC^2 / 2
sqrt2OmegaChi = sqrt(2pho.omegaC) * pho.chi
chi2OverOmega  = pho.chi^2 / pho.omegaC
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
#     qMolMax = 1.5x[1]
#     qPhoMax = 3x[2]
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
#     qMol = LinRange(-qMolMax, qMolMax, 101)
#     qPho = LinRange(-qPhoMax, qPhoMax, 101)
#     mep = LinearInterpolation([-qMolMax, qMolMax], [-qPhoMax/2, qPhoMax/2])
# 
#     for i in qMol
#         y = mep(i)
#         u = uTotal(i, y, halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
#         @printf(mepOut, "%5.2f %5.2f %9.6f \n", i, y, u)
#         @printf(mepOut, "\n")
#         for j in qPho
#             u = uTotal(i, j, halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
#             @printf(pesOut, "%5.2f %5.2f %9.6f \n", i, j, u)
#         end
#         @printf(pesOut, "\n")
#     end
# 
# end
 
chi = [0.002]
omegaC = [pho.omegaC]
a_bc2 = collect(0:100)
f = open("omegac", "w")
# @printf(f, "# ω ")
# for i in omegaC
#     @printf(f, "%8.3f ", i*consts.au2wn)
# end
# @printf(f, "\n")

for j in chi
    for i in omegaC
        for k in a_bc2
#         halfOmegaC2 = i^2 / 2
#         sqrt2OmegaChi = sqrt(2i) * j
#         chi2OverOmega  = j^2 / i
#         getImFreq(halfOmegaC2, sqrt2OmegaChi, chi2OverOmega)
            tmp = solve(i, j, k)
            @printf(f, "%8.3f %8.3f \n", k.au2wn, tmp[1])
        end
    end
    @printf(f, "\n")
end
