using DelimitedFiles
using JLD
using LsqFit
using Statistics: mean
# using PyPlot

function getCoefficients(pes::T1="pes.txt", dm::T1="dm.txt";
    coeffFile::Union{T1, T2}=nothing) where {T1<:String, T2<:Nothing}
    coefficients = Dict{String, Union{Matrix{Float64}, Float64}}()
    if coeffFile !== nothing
        if occursin("jld", coeffFile)
            jldopen(coeffFile) do io
                coefficients = read(io)
            end
        else
            throw("Only `jld` input is supported currently.")
        end
        return coefficients["dm"], coefficients["a0"], coefficients["pes"]
    end
    raw, order = fitData(dm, :sine, minOrder=5, convergence=7e-4)
    raw = reshape(raw, 3, order)
    @views raw[1, :] /= 2.0
    dmCoeff = vcat(raw, (-raw[1, :].*raw[3, :])')
    push!(coefficients, "dm" => dmCoeff)
    raw, order = fitData(pes, :fourierCosine, convergence=3e-5, minOrder=8)
    a0 = raw[1]
    w = raw[2] / 2.0
    a = @view raw[3:end]
    potCoeff = Matrix{Float64}(undef, 3, length(a))
    for i in eachindex(1:length(a))
        iw = i * w
        potCoeff[1, i] = iw
        potCoeff[2, i] = a[i]
        potCoeff[3, i] = a[i] * iw
    end
    push!(coefficients, "a0" => a0)
    push!(coefficients, "pes" => potCoeff)
    save("coefficients.jld", coefficients, compress=true)
    return dmCoeff, a0, potCoeff
end

function fourierCosine(x::AbstractVector{T}, a::AbstractVector{T}) where T<:Real
    u = repeat([a[1]], length(x))
    for i in 3:length(a)
        @. u += a[i] * cos((i-2) * a[2] * x / 2.0)
    end
    return u
end

function sine(x::AbstractVector{T}, a::AbstractVector{T}) where T<:Real
    u = repeat([0.0], length(x))
    for i in 1:convert(Int64, length(a)/3)
        @. u += a[3i] * sin(a[3i-2] * x / 2.0 + a[3i-1])
    end
    return u
end

function fitData(file::String, model::Symbol; slice=1000:8999,
    convergence=2e-4, minOrder=6, maxOrder=10)

    raw = readdlm(file)
    xdata = @views raw[slice, 1] * 2.0
    ydata = @views raw[slice, 2] / 2.0
    inputX = @view xdata[1:10:end]
    exactY = @view ydata[1:10:end]

    if model == :sine
        # parameters = ones(3*minOrder)
        n0 = 3 * minOrder
        increment = 3
    else
        n0 = minOrder + 2
        # parameters = ones(minOrder+2)
        increment = 1
    end
    func = getfield(Main, model)
    w = repeat([0.3], length(xdata))
    @views w[2000:6000] .= 1.0
    for i in minOrder:maxOrder
        fitted = curve_fit(func, xdata, ydata, repeat([1.0], n0))
        fitValue = func(inputX, fitted.param)
        averageError = mean(abs.(fitValue .- exactY))
        # writedlm("log", [inputX fitValue.-exactY])
        # println(averageError)
        # plot(inputX, fitValue)
        # plot(inputX, exactY)
        # PyPlot.show()
        if averageError < convergence
            return fitted.param, i
        end
        n0 += increment
        # parameters = vcat(parameters, ones(increment))
    end
    throw("Could not meet convergence = $convergence within $maxOrder orders.")
end