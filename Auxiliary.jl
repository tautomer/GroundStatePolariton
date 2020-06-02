module Auxiliary
"""
    Some physical constants useful to the simulation
"""
module Constants

export au2wn, au2ev, au2kelvin, au2kcal, amu2au, au2fs

au2wn = 219474.63068
au2ev = 27.2113961
au2kelvin = 3.15775e5
au2kcal = 627.509
amu2au = 1836.0
au2fs = 2.41888e-2

end

push!(LOAD_PATH, "./")
using .Constants
using Calculus

"""
    normalModeAnalysis(x0::Array{Float64,1})

Return normal mode frequencies, eigenvectors and crossover temperature
"""
function normalModeAnalysis(u::Function, x0::Array{Float64,1}, omegaC::Float64)
    hf = Calculus.hessian(x -> u(x[1], x[2]))
    mat = hf(x0)
    mat[1, 1] /= amu2au
    mat[2, 1] /= sqrt(amu2au)
    mat[1, 2] = mat[2, 1]
    # possibly a bug in Calculus package
    # when ω is fairly small, ∂²V/∂q² will be ZERO
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

end