#!/bin/bash

# Define arrays with given values
M_vals=(4 8 8 8 8 32 32 32 32 8 16 32 64 128)
K_vals=(8 16 32 64 128 16 32 64 128 8 16 32 64 128)
N_vals=(4 8 16 16 32 8 16 16 32 8 16 32 64 128)
m_vals=(4 4 4 4 4 4 4 4 4 2 2 2 2 2)
k_vals=(8 8 8 8 8 8 8 8 8 8 8 8 8 8)
n_vals=(4 4 4 4 4 4 4 4 4 8 8 8 8 8)

# Iterate through the arrays and call make
for i in "${!M_vals[@]}"; do
    M=${M_vals[$i]}
    K=${K_vals[$i]}
    N=${N_vals[$i]}
    m=${m_vals[$i]}
    k=${k_vals[$i]}
    n=${n_vals[$i]}

    echo "Running make with: i=$i M=$M K=$K N=$N m=$m k=$k n=$n"
    make clean && make i=$i M=$M K=$K N=$N m=$m k=$k n=$n run_sim

done
