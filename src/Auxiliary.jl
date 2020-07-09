module Auxiliary
"""
    Some physical constants useful to the simulation
"""

using Constants
using IntervalArithmetic, IntervalRootFinding

# FIXME: switch to packages like FiniteDiff or ForwardDiff
"""
    normalModeAnalysis(x0::Array{Float64,1})

Return normal mode frequencies, eigenvectors and crossover temperature
"""
function normalModeAnalysis(u::Function, x0::Array{Float64,1}, omegaC::Float64)
    hf = Calculus.hessian(x -> u(x[1], x[2]))
    mat = hf(x0)
    # possibly a bug in Calculus package
    # when ω is fairly small, ∂²V/∂q² will be ZERO
    mat_og = copy(mat)
    mat[1, 1] /= amu2au
    mat[2, 1] /= sqrt(amu2au)
    mat[1, 2] = mat[2, 1]
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

"""
    function saddlePoints(f::Function, low::T, high::T) where T<:Real

Find the roots of a given univariate function f in a given range [low, high],
and return a sorted vector of roots via `IntervalRootFinding`. Note that only 
a pure Julia function will work with `IntervalRootFinding`.
"""
function saddlePoints(f::Function, low::T, high::T) where T<:Real
    result = roots(f, low..high)
    loc = Vector{Float64}(undef, 0)
    for i in result
        push!(loc, mid(i.interval))
    end
    return sort(loc)
end

end