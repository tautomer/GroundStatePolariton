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
    input = KappaInput(3, 1, 1, 1, 2, 300.0, wc, chi, :systemBath, :fullSystem,
        :ordered, :twoBarriers) 
    @time computeKappa(input)
    input.ntraj = 300000
    input.nstep = 2000
    Profile.clear_malloc_data()
    @time computeKappa(input)
end

function rpmdrate(wc::Real, chi::Real, method::Symbol)
    cd("rate/4.0")
    if length(ARGS) != 0
        nb = parse(Int64, ARGS[1])
        wc = parse(Float64, ARGS[2])
        chi = parse(Float64, ARGS[3])
        temp = parse(Float64, ARGS[4])
    else
        wc = wc
        chi = chi
        temp = 300.0
        nb = 1
    end
    input = UmbrellaInput(2, nb, 2, 1, temp, 0.12, [-3.2, 3.2], wc, chi,
        method)
    umbrellaSampling(input)
    input.nw = 31
    input.nstep = convert(Int64, 5e6)
    xbin, pmf = umbrellaSampling(input)
    input = KappaInput(2, nb, 1, 1, temp, wc, chi, :systemBath, :fullSystem,
        :ordered) 
    computeKappa(input)
    input.ntraj = 200_000
    input.nstep = 2000
    fs = computeKappa(input)
    tst = WHAM.TSTRate(xbin, pmf, au2kelvin/j, amu2au)
    flnm = string("rate_", wc, "_", chi, "_", temp, "_", nb, ".txt")
    out = open(flnm, "w")
    k_in_au = rate(out, fs, tst, au2kelvin/j)
    close(out)
end

function read()
    ev2k =1.160452e4
    wc = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.032,
        0.04, 0.06, 0.08, 0.12, 0.16, 0.2]
    eta = 2.0 
    temp = 300.0
    cd("rate")
    out = string("rate_", eta, "_", temp, ".txt")
    rateout = open(out, "w")
    for i in wc
        println(i)
    	w = i * ev2k
	chi = i * eta
    	order = floor(log2(w/temp))
    	if order <= 0
    	    nb = 32
    	else
    	    nb = convert(Int64, 2^(order+4))
    	end
    	fsout = string("fs_", i, "_", chi, "_", temp, "_", nb, ".txt")
	fs = readdlm(fsout, comments=true)
	@views kappa = fs[:, 2]
    	pmfout = string("pmf_", i, "_", chi, "_", temp, "_", nb, ".txt")
	pmf = readdlm(pmfout, comments=true)
	@views bin = pmf[:, 1]
	@views f = pmf[:, 2]
    	tst = WHAM.TSTRate(bin, f, au2kelvin/temp, amu2au)
    	k_in_au = rate(rateout, kappa, tst, au2kelvin/temp)
    end
    close(rateout)
end

function testPMF(wc::Real, chi::Real, method::Symbol)
    cd("energy")
    if length(ARGS) != 0
        nb = parse(Int64, ARGS[1])
        wc = parse(Float64, ARGS[2])
        chi = parse(Float64, ARGS[3])
        temp = parse(Float64, ARGS[4])
    else
        wc = wc
        chi = chi
        temp = 300.0
        nb = 1
    end
    input = UmbrellaInput(3, nb, 2, 1, 2, temp, 0.3, [-3.5, 3.5], wc, chi,
        method, :twoBarriers) 
        # method, :twoBarriers) 
    @time umbrellaSampling(input)
    input.nw = 41
    input.nstep = convert(Int64, 1e8)
    Profile.clear_malloc_data()
    @time bin, f = umbrellaSampling(input)
    print(WHAM.TSTRate(bin, f, au2kelvin/temp, amu2au))
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

function resonance(η::Float64, np::Integer, constrined::Integer)
    # wc = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.032,
    #     0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.4, 0.6, 0.8, 1.0, 3.0, 5.0]
    wc = [0.00875, 0.015, 0.02, 0.0285, 0.036, 0.044, 0.048, 0.052, 0.056,
        0.07, 0.1, 0.14, 0.18]
    kappa = similar(wc)
    #input = KappaInput(2, 1, 1, 1, 300.0, wc[1], η*wc[1], :systemBath, :fullSystem,
    #    :ordered) 
    input = KappaInput(np, 1, 1, 1, constrined, 300.0, wc[1], η*wc[1], :systemBath, :fullSystem,
        :ordered, :twoBarriers) 
    # input = InputValues(np, 1, 1, 300.0, wc[1], η*wc[1], :langevin,
    #     :fullSystem, :ordered) 
    # dir = "scaneta"
    dir = string(η, "_", np-1, "_", constrined)
    if ! isdir(dir)
        mkdir(dir)
    end
    cd(dir)
    computeKappa(input)
    input.nstep = 2000
    input.ntraj = 200000
    Threads.@threads for i in eachindex(wc)
        χ = η * wc[i]
        if length("$χ") >= 10
            χ = round(χ, digits=9)
        end
        input.ωc = wc[i]
        # input.ωc = η
        input.χ = χ
        fs = computeKappa(input)
        @views kappa[i] = mean(fs[end-50:end])
    end
    writedlm("kappa", [wc kappa])
end

function scaneta(wc::Float64, np::Integer, constrined::Integer)
    eta = [0.0001, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    kappa = similar(eta)
    input = KappaInput(np, 1, 1, 1, constrined, 300.0, wc, eta[1]*wc, :systemBath, :fullSystem,
        :ordered, :twoBarriers) 
    # dir = "scaneta"
    dir = string("wc_", wc, "_", np-1, "_", constrined)
    if ! isdir(dir)
        mkdir(dir)
    end
    cd(dir)
    computeKappa(input)
    input.nstep = 5000
    input.ntraj = 200000
    Threads.@threads for i in eachindex(eta)
        χ = wc * eta[i]
        input.χ = χ
        fs = computeKappa(input)
        @views kappa[i] = mean(fs[end-50:end])
    end
    writedlm("kappa", [eta/sqrt(amu2au) kappa])
end

function scan()
    ωc = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.032,
        0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.4, 0.6, 0.8, 1.0]
    η = [0.0001, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    tst = [1.3917051728344806e-23, 1.6557290752653633e-23]
    rate = similar(tst)
    iter = [(i, j, k) for i in ωc, j in η, k in [1, 2]]
    κ = Array{Float64}(undef, size(iter))
    dir = "scan2d"
    if ! isdir(dir)
        mkdir(dir)
    end
    cd(dir)
    input = repeat([KappaInput(3, 1, 1, 1, 1, 300.0, 10.0, η[1]*ωc[1],
        :systemBath, :fullSystem, :ordered, :twoBarriers)], Threads.nthreads()) 
    computeKappa(input[1])
    for i in input
        i.nstep = 2000
        i.ntraj = 200000        
    end
    # for i in eachindex(iter)
    Threads.@threads for i in eachindex(iter)
        tid = Threads.threadid()
        inp = input[tid]
        config = iter[i]
        inp.ωc = config[1]
        χ = inp.ωc * config[2]
        if length("$χ") >= 10
            inp.χ = round(χ, digits=9)
        else
            inp.χ = χ
        end
        inp.constrained = config[3]
        flnm = string("fs_", inp.ωc, "_", inp.χ, "_", 300.0, "_", inp.nb,
            "_", inp.constrained, ".txt")
        if isfile(flnm)
	        fs = readdlm(flnm, comments=true)
        else
            println("File ", flnm, " not exist")
            fs = computeKappa(inp)
        end
        @views κ[i] = mean(fs[end-50:end])
    end
    out = open("yield", "w")
    for i in eachindex(ωc)
        for j in eachindex(η)
            @printf(out, "%7.4f %7.4f ", ωc[i], η[j])
            for k in [1, 2]
               rate[k] = κ[i, j, k] * tst[k]
               @printf(out, "%9.7f ", κ[i, j, k])
            end
            k_total = sum(rate)
            @printf(out, "%11.7g %11.7g %9.7f %9.7f\n", rate[1], rate[2], rate[1]/k_total, rate[2]/k_total)
        end
        @printf(out, "\n")
    end
end

resonance(2.0, 3, 1)
# scaneta(0.025, 3, 1)
# computeΔΔG("data/eta_scan.txt", 300.0)
# testKappa(0.025, 2.0)
# testPMF(0.064, 0.064, :UI)
# rpmdrate(0.005, 0.04, :UI)
# read()
# scan()
