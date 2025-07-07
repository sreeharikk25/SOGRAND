SOGRAND for Cubic Codes - Pure C Implementation This project is a standalone C-language version of the sim_cubic.m MATLAB simulation. It allows for high-speed simulation of the SOGRAND algorithm for cubic tensor product codes without requiring a MATLAB license.

How to Run

Compilation This program is written in standard C and requires the math library. To compile it, open a terminal in the source directory and run:
gcc sogrand_cubic_sim.c -o sogrand_cubic_sim -lm

Execution After successful compilation, an executable file named sogrand_cubic_sim will be created. Run it from your terminal with the following command:
./sogrand_cubic_sim

The program will then start the Monte Carlo simulation and print the results for each Eb/N0 value to the console.
