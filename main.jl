push!(LOAD_PATH, "./")
using ImportMacros
using DelimitedFiles
using Dierckx
using Printf
using Random
using Dynamics
@import CorrelationFunctions as corr


function run()
    nSteps = 1000
    nTraj = 10000
    mass = 1836.0
    temp = 300 / 3.15775e5
    omegaC = 0.2 / 27.2113961
    freqCutoff = 500 / 219474.63068
    eta = 0.5 * mass * freqCutoff
    chi = 0.02

    param = Parameters(temp, 10.0, 1, 0, 15, 1, 1, 1)

    mol = [ClassicalParticle("mol", mass, 0.0, 0.0) for i in 1:param.nMol]
    if param.nPho != 0
        push!(mol, ClassicalParticle("photon", 1.0, 0.0, 0.0))
    end
    ω = Vector{Float64}(undef, param.nBath)
    for i in 1:param.nBath
        ω[i] = -freqCutoff * log((i-0.5) / param.nBath)
    end
    c = ω * sqrt(2eta * mass * freqCutoff / param.nBath / pi)
    mω2 = mass * ω.^2
    c_mω2 = c ./ mω2
    bath = ClassicalBathMode(param.nBath, mass, ω, c, mω2, c_mω2, c, c)

    potentialRaw = readdlm("pes.txt")
    dipoleRaw = readdlm("dm.txt")
    pesMol = Spline1D(potentialRaw[:, 1]*1.5, potentialRaw[:, 2])
    dipole = Spline1D(dipoleRaw[:, 1], dipoleRaw[:, 2])
    # photon DOF potential
    pesPho(qPho) = 0.5 * omegaC^2 * qPho^2
    # light-mater interaction
    lightMatter(qMol, qPho) = sqrt(2*omegaC)*chi*dipole(qMol)*qPho +
        chi^2*dipole(qMol)^2/omegaC
    # total potential
    uTotal(qMol, qPho) = pesMol(qMol) + pesPho(qPho) + lightMatter(qMol, qPho)

    
    fs0 = 0.0
    fs = zeros(nSteps+1)
    for i in 1:nTraj
        mol[1].x = 0.0
        mol[1].v = Dynamics.velocitySampling(param, mol[1])
        bath.v = Dynamics.velocitySampling(param, bath)
        bath.x = Random.randn(bath.n)
        fc = Dynamics.force(pesMol, mol[1].x, bath)
        for i in 1:100
            mol[1], bath = Dynamics.velocityUpdate(param, fc, mol[1], bath)
            mol[1], bath = Dynamics.positionUpdate(param, fc, mol[1], bath, cnstr=true)
            fc = Dynamics.force(pesMol, mol[1].x, bath)
            mol[1], bath = Dynamics.velocityUpdate(param, fc, mol[1], bath)
            if i % 10 == 0
                mol[1].v = Dynamics.velocitySampling(param, mol[1])
                bath.v = Dynamics.velocitySampling(param, bath)
            end
        end
        q = zeros(nSteps+1)
        v0 = mol[1].v
        fs0 = corr.fluxSide(fs0, v0, v0)
        for j in 1:nSteps
            mol[1], bath = Dynamics.velocityUpdate(param, fc, mol[1], bath)
            mol[1], bath = Dynamics.positionUpdate(param, fc, mol[1], bath)
            fc = Dynamics.force(pesMol, mol[1].x, bath)
            mol[1], bath = Dynamics.velocityUpdate(param, fc, mol[1], bath)
            q[j+1] = mol[1].x
            # println(j, " ", mol[1].v, " ", q[j+1], " ", 0.5*mass*v^2+pesMol(q[j+1]))
            # println(j, " ", mol[1].v, " ", q[j+1])
        end
        fs = corr.fluxSide(fs, v0, q)
    end

    fs /= fs0
    fs[1] = 1.0
    fsOut = open("fs.txt", "w")
    for i in 1:nSteps+1
        @printf(fsOut, "%5.2f %5.3f\n", i*param.Δt*2.4189e-2, fs[i])
    end
end

run()
