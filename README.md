# A Julia program for ground-state polaritonic chemistry model system

100% Julia implementation.

PES and the permenant dipole is generated from the Shin-Metiu model and then
fitted with a Fourier series (for PES) and sine series (for DM). 

This program currently handles single-molecule, many-molecule and single-photon case in both classical and quantum (via RPMD) treatment.
The transimission coefficient and free energy barrier are computed separately.
Then the rate is calculated from these values.

The source files are in the `src` folder. All files other than `Polariton.jl`
are legacy. I'm not sure how to deal with them, but ultimately I would like to
split the `Polariton.jl` into different files. Right now it's code structure is
like a piece of shit.

## TODO list

1. 2 different barriers (WIP)
2. File structure improvement
3. More sensible input parameters, pabably from a JSON file?
4. Debugging options. Use multiple dispatch?