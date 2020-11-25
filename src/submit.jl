const ev2k =1.160452e4
const wc = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.032,
    0.04, 0.06, 0.08, 0.12, 0.16, 0.2, 0.4, 0.6, 0.8]
const temp = 300.0

for i in wc
    w = i * ev2k
    order = floor(log2(w/temp))
    if order <= 0
        nb = 32
    else
        nb = convert(Int64, 2^(order+4))
    end
    out = open("sbatch", "w")
    println(out, "#!/bin/bash
#SBATCH -p action -A action
#SBATCH -N 1
#SBATCH --mem-per-cpu=10gb
#SBATCH -J 4.0_", i, "
#SBATCH -o runtime_2.0_", i, "
#SBATCH -t 5-00:00:00

julia Polariton.jl ", nb, " ", i, " ", 4.0*i, " ", temp)
    close(out)
    run(`sbatch sbatch`)
end
