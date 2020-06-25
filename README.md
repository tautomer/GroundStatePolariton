# A Julia program for ground-state polaritonic chemistry model system

100% Julia implementation.

PES and the permenant dipole is generated from the Shin-Metui model and then
read from text file. I'm using linear interpolation because it's cheaper. By
using more points in the originl Shin-metui script, the linear interpolated
PES, in fact, has higher quality.

This program currently handles single-molecule and single-photon case.
The transimission coefficient and free energy barrier are computed separately.
Then the rate is calculated from these values.

The source files are in the `src` folder. All files other than `Polariton.jl`
are legacy. I'm not sure how to deal with them, but ultimately I would like to
split the `Polariton.jl` into different files. Right now it's code structure is
like a piece of shit.

## TODO list

1. Multi-molecule (WIP)
2. RPMD
3. File structure improvement
4. More sensible input parameters, pabably from a JSON file?
5. Debugging options. Use multiple dispatch?