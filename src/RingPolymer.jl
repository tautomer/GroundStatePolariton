using LinearAlgebra: eigvals, eigvecs

struct RPMDArrays
    ωₖ::Vector{Float64}
    tnm::Matrix{Float64}
    tnmi::Matrix{Float64}
    freerp::Matrix{Float64}
end

function ringPolymerSetup(nb::Integer, dt::T, temp::T, mass::T) where T<:Real
    # ωₙ = 1 / βₙ
    ωₙ = nb * temp
    ωₖ, tnm, tnmi = ringNormalMode(nb)
    ωₖ *= ωₙ
    freerp = getFreeRPMatrix(nb, dt, mass, ωₖ)
    rpmd = RPMDArrays(ωₖ, tnm, tnmi, freerp)
    return rpmd
end

"""
    function ringNormalMode(nb::Integer)

Function to compute the normal mode frequencies and transformation matrices.
All values can also be computed analytically. 

    twoPiByN = 2pi / nb
    sqrtInvN = sqrt(1.0/nb)
    sqrtInv2oN = sqrt(2.0/nb)
    ωₖ = Vector{Float64}(undef, nb)
    tnmi = Matrix{Float64}(undef, nb, nb)
    if nb % 2 == 0
        halfN = nb ÷ 2
        for i in 1:nb
            tnmi[i, halfN+1] = sqrtInvN * (-1)^i
        end
    else
        halfN = nb ÷ 2 + 1
    end
    for i in 1:nb
        tmp = twoPiByN * i
        ωₖ[i] = 2 * sin((i-1)*pi/nb)
        tnmi[i, 1] = sqrtInvN
        for j in 2:halfN
            tnmi[i, j] = sqrtInv2oN * cos(tmp*(j-1))
        end
        for j in halfN+2:nb
            # println(i, " ", j, " ", sin(tmp*(j-1)))
            tnmi[i, j] = sqrtInv2oN * sin(tmp*(j-1))
        end
    end
    tnm = tnmi'
"""
function ringNormalMode(nb::Integer)
    # build the matrix
    m = zeros(nb, nb)
    m[1, 1] = 2.0
    m[1, 2] = -1.0
    m[1, nb] = -1.0
    for i in 2:nb-1
        m[i, i-1] = -1.0
        m[i, i] = 2.0
        m[i, i+1] = -1.0
    end
    m[nb, 1] = -1.0
    m[nb, nb-1] = -1.0
    m[nb, nb] = 2.0
    # get ωₖ's (note: ωₙ hasn't been included here)
    ωₖ = sqrt.(eigvals(m))
    # get inverse normal mode transformation matrix
    tnmi = eigvecs(m)
    # `DEPEV` can give wrong phase
    if tnmi[1, 1] < 0
        tnmi .*= -1.0
    end
    # forward NM transformation matrix
    tnm = tnmi'
    return ωₖ, tnm, tnmi
end

function getFreeRPMatrix(nb::Integer, dt::T, mass::T, ω::AbstractVector{T}
    ) where T<:AbstractFloat

    a = Matrix{Float64}(undef, 4, nb)
    # centroid
    a[1, 1] = 1.0
    a[2, 1] = 0.0
    a[3, 1] = dt / mass
    a[4, 1] = 1.0
    # other modes
    for i in 2:nb
        ωi = ω[i]
        mω = mass * ωi
        ωt = ωi * dt
        sinωt = sin(ωt)
        cosωt = cos(ωt)
        a[1, i] = cosωt
        a[2, i] = -mω * sinωt
        a[3, i] = sinωt / mω
        a[4, i] = cosωt
    end
    return a
end

temp = 300.0 / 315775.0
dt = 2.0
mass = 1836.0
rpmd = ringPolymerSetup(4, dt, temp, mass)
freq = 1500.0 / 219474.63068
force(x) = -0.08576034929437469 * x

using Statistics: mean
x = [1.0, 1.1, 0.9, 1.2]
v = [5e-4, 6e-4, 7e-4, 9e-4]
v .*= mass
f = force.(x)
xt = copy(x)
vt = copy(v)
vout = open("v", "w")
xout = open("x", "w")
for i in 1:10000
    @. v += 0.5 * f * dt
    xt .= rpmd.tnm * x
    vt .= rpmd.tnm * v
    for j in 1:4
        newvt = rpmd.freerp[1, j] * vt[j] + rpmd.freerp[2, j] * xt[j]
        xt[j] = rpmd.freerp[3, j] * vt[j] + rpmd.freerp[4, j] * xt[j]
        vt[j] = newvt
    end
    x .= rpmd.tnmi * xt
    v .= rpmd.tnmi * vt
    @. f = force(x)
    @. v += 0.5 * f * dt
    println(vout, i, " ", mean(v), " ", v[1], " ", v[2], " ", v[3], " ", v[4])
    println(xout, i, " ", mean(x), " ", x[1], " ", x[2], " ", x[3], " ", x[4])
end
close(vout)
close(xout)