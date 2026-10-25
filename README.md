# SymNMF Clustering

Symmetric Non-negative Matrix Factorization (SymNMF) clustering implemented in C, with a Python C extension so it can be called from Python.

## What it does

Given n data points, the program builds:

1. **Similarity matrix** `A`: `a_ij = exp(-||x_i - x_j||^2 / 2)`, with zeros on the diagonal
2. **Diagonal degree matrix** `D`: `d_ii = sum_j a_ij`
3. **Normalized similarity matrix** `W = D^(-1/2) A D^(-1/2)`

It then iteratively updates `H` (n x k) to minimize `||W - HH^T||^2` and assigns each point to the cluster with the largest value in its row of `H`.

## Project structure

| File | Purpose |
|---|---|
| `symnmf.c` / `symnmf.h` | Core algorithm and C command-line entry point |
| `matrix_util.c` / `matrix_util.h` | Matrix operations and memory pool |
| `parse_input.c` / `parse_input.h` | Input file parsing |
| `symnmfmodule.c` | Python C extension bindings |
| `symnmf.py` | Python interface |
| `kmeans.py`, `analysis.py` | K-means baseline and silhouette-score comparison |

## Build

```bash
python3 setup.py build_ext --inplace   # Python extension
make                                   # C executable
```

Developed and tested with gcc on Ubuntu. The Makefile uses strict flags (`-ansi -Wall -Wextra -Werror -pedantic-errors`).

## Usage

```bash
# goal: sym | ddg | norm | symnmf
python3 symnmf.py <k> <goal> <input_file>
./symnmf <goal> <input_file>             # sym | ddg | norm

# compare SymNMF and k-means by silhouette score
python3 analysis.py <k> <input_file>
```

Input files contain one comma-separated data point per line (see `input_1.txt`).
