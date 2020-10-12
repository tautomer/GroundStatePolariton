include("Tasks.jl")

cd("..")
# const dipoleCoeff, v0, pesCoeff = getCoefficients(coeffFile="coefficients.jld")
# const dipoleCoeff, v0, pesCoeff = getCoefficients()

function temperatureDependency()
    cd("tempDep")
    nTraj = 10000
    nStep = 3000
    temp = collect(233.0:15.0:413.0)
    # temp = [233.0]
    # omegac = vcat(collect(0.01:0.01:0.03), collect(0.1:0.1:0.3))
    omegac = [0.16]
    # chi = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    # iter = vcat([(0.16, 0.16i) for i in chi])
    # # iter = reduce(vcat, [[(i,0.1i), (i,0.3i)] for i in omegac])
    iter = [(0.16, 0.0)]
    cd("tmp")
    computeKappa(2, 300.0, 1, 1, omegac[1], omegac[1])
    cd("..")

    pmf = readdlm("pmf_0.2_0.0_300.0.txt", comments=true)
    # xbin, pmf = umbrellaSampling(temp[end], 100, 0.15, [-3.5, 3.5], 0.2, 0.0)
    Threads.@threads for i in iter
        ωc = i[1]
        χ = i[2]
        if length("$χ") >= 10
            χ -= eps(χ)
        end
        flnm = string("rate_", ωc, "_", χ, ".txt")
        output = open(flnm, "w")
        @printf(output, "# ω_c=%5.3f,χ=%6.4f \n", ωc, χ)
        @printf(output, "# Thread ID: %3d\n", Threads.threadid())
        @printf(output, "# Warning: check κ mannually if line ends with 'flagged'\n")
        @printf(output, "# 1/T    log k/au    log k/T     T      log k/s     k(au)       k(s^-1)      κ        TST(au)     Flag\n")
        @time for j in temp
            println("Currently running ωc = $ωc, χ = $χ, T = $j")
            fs = computeKappa(2, j, nTraj, nStep, ωc, χ)
            tst = WHAM.TSTRate(pmf[:, 1], pmf[:, 2], au2kelvin/j, amu2au)
            k = rate(output, fs, tst, au2kelvin/j)
        end
    end
end

# temperatureDependency()
# dir = "test"
# if ! isdir(dir)
#     mkdir(dir)
# end
# cd(dir)
using Profile
function testKappa(wc, eta)
    cd("energy")
    chi = eta * wc
    # input = InputValues(2, 10, 5000, 300.0, wc, chi, :langevin, :fullSystem,
    input = InputValues(2, 48, 1, 1, 300.0, wc, chi, :systemBath, :fullSystem,
        :ordered) 
    @time computeKappa(input)
    input.ntraj = 1000
    input.nstep = 1000
    Profile.clear_malloc_data()
    @time computeKappa(input)
end

function testPMF()
    @time umbrellaSampling(300.0, 100, 10, 0.15, [-3.5, 3.5], 0.16, 0.0)
    Profile.clear_malloc_data()
    @time umbrellaSampling(300.0, 100, 10000000, 0.15, [-3.5, 3.5], 0.16, 0.0)
end

function computeΔΔG(kappaFile::String, temp::Real)
    β = au2kelvin / temp
    data = readdlm(kappaFile) 
    dG = @view data[:, 2]
    @. dG = -1.0 / β * log(2pi * β * dG)
    for i in 2:length(dG)
        dG[i] -= dG[1]
    end
    dG[1] = 0.0
    writedlm("data/ddg.txt", data)
end

function resonance(η::Float64, np::Integer)
    wc = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.032,
        0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.4, 0.6, 0.8, 1.0, 3.0, 5.0]
    # wc = [0.0025, 0.0075, 0.025]
    # wc = [0.0001, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    kappa = similar(wc)
    input = InputValues(np, 1, 1, 300.0, wc[1], η*wc[1], :langevin,
        :fullSystem, :ordered) 
    # dir = "scaneta"
    dir = string(η, "_", np-1, "_no_dse")
    if ! isdir(dir)
        mkdir(dir)
    end
    cd(dir)
    computeKappa(input)
    input.nstep = 2000
    input.ntraj = 10000
    Threads.@threads for i in eachindex(wc)
        χ = η * wc[i]
        if length("$χ") >= 10
            χ -= eps(χ)
        end
        input.ωc = wc[i]
        # input.ωc = η
        input.χ = χ
        fs = computeKappa(input)
        @views kappa[i] = mean(fs[end-50:end])
    end
    writedlm("kappa_new", [wc kappa])
end

function scaneta(wc::Float64, np::Integer)
    eta = [0.0001, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    kappa = similar(eta)
    input = InputValues(np, 1, 1, 300.0, wc, eta[1]*wc, :langevin,
        :fullSystem, :ordered) 
    # dir = "scaneta"
    dir = string("wc_", wc, "_", np-1, "")
    if ! isdir(dir)
        mkdir(dir)
    end
    cd(dir)
    computeKappa(input)
    input.nstep = 2000
    input.ntraj = 100000
    Threads.@threads for i in eachindex(eta)
        χ = wc * eta[i]
        input.χ = χ
        fs = computeKappa(input)
        @views kappa[i] = mean(fs[end-50:end])
    end
    writedlm("kappa", [eta/sqrt(amu2au) kappa])
end

# resonance(0.01, 2)
# scaneta(0.1706, 2)
# computeΔΔG("data/eta_scan.txt", 300.0)
testKappa(1.0, 1.0)