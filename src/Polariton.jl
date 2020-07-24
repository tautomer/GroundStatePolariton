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
    # iter = reduce(vcat, [[(i,0.1i), (i,0.3i)] for i in omegac])
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
dir = "test"
if ! isdir(dir)
    mkdir(dir)
end
cd(dir)
using Profile
function testKappa()
    if length(ARGS) != 0
        np = parse(Int64, ARGS[1])
        wc = parse(Float64, ARGS[2])
    else
        np = 2
        wc = 0.17
    end
    chi = 0.5 * wc # / sqrt(np)
    input = InputValues(np, 1, 1, 300.0, wc, chi, :langevin, :fullSystem,
    # input = InputValues(np, 1, 1, 300.0, wc, chi, :langevin, :normalModes,
        :ordered) 
    @time computeKappa(input)
    input.ntraj = 1000
    input.nstep = 2000
    Profile.clear_malloc_data()
    @time computeKappa(input)
end

function testPMF()
    @time umbrellaSampling(300.0, 100, 10, 0.15, [-3.5, 3.5], 0.16, 0.0)
    Profile.clear_malloc_data()
    @time umbrellaSampling(300.0, 100, 10000000, 0.15, [-3.5, 3.5], 0.16, 0.0)
end

function resonance(η::Float64, np::Integer)
    wc = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.08, 0.12, 0.16, 0.2, 0.4,
        0.6, 0.8, 1.0, 3.0, 5.0]
    # wc = [0.0005, 0.001, 0.005, 0.01, 0.04, 0.08, 0.12, 0.16, 0.2, 0.4, 0.6,
    #     0.8, 1.0]
    kappa = similar(wc)
    input = InputValues(np, 1, 1, 300.0, wc[1], 0.05*wc[1], :langevin,
        :fullSystem, :ordered) 
    # dir = "scaneta"
    dir = string(η, "_", np-1, "")
    if ! isdir(dir)
        mkdir(dir)
    end
    cd(dir)
    computeKappa(input)
    input.nstep = 10000
    input.ntraj = 3000
    Threads.@threads for i in eachindex(wc)
        χ = eta * wc[i]
        if length("$χ") >= 10
            χ -= eps(χ)
        end
        input.ωc = wc[i]
        input.ωc = eta
        input.χ = χ
        fs = computeKappa(input)
        @views kappa[i] = mean(fs[end-50:end])
    end
    writedlm("kappa", [wc kappa])
end

resonance(0.5, 2)
# testKappa()