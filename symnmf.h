#ifndef SYMNMF_H
#define SYMNMF_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix_util.h"
#include "parse_input.h"
/*
Performs one SymNMF update step on matrix H using the current H and W.

Input:
  H (MatrixPtr)- current factor matrix (n x k)
  W (MatrixPtr)- normalized similarity matrix (n x n)

Output:
  Returns a newly allocated updated H matrix,
  or NULL on error.
*/
MatrixPtr updateH(MatrixPtr H, MatrixPtr W);

/*
Computes the difference H_updated - H and checks whether
its squared Frobenius norm is below epsilon.

Input:
  H          - previous matrix (will be modified)
  H_updated  - new matrix
  epsilon    - convergence threshold

Output:
  Returns 1 if converged, 0 otherwise.
*/
int checkConvergence(MatrixPtr H, MatrixPtr H_updated, double epsilon);

/*
Runs the iterative SymNMF process starting from H
until convergence or stopping condition is reached.

Input:
  H (MatrixPtr)- initial factor matrix (n x k)
  W (MatrixPtr)- normalized similarity matrix (n x n)

Output:
  Returns the final optimized H matrix,
  or NULL on error.
*/
MatrixPtr getResultH(MatrixPtr H, MatrixPtr W);

/*
Builds the similarity matrix A from the data matrix X.

For every pair of rows in X, computes their similarity score
according to the assignment formula.

Input:
  X (MatrixPtr)- data matrix where each row is a data point (n x d)

Output:
  Returns the similarity matrix A (n x n),
  or NULL on error.
*/
MatrixPtr getSimilarityMatrix(MatrixPtr X);

/*
Builds the diagonal degree matrix D from similarity matrix A.

Each diagonal entry D[i][i] equals the sum of row i in A.
All off-diagonal entries are zero.

Input:
  A (MatrixPtr)- similarity matrix (n x n)

Output:
  Returns the diagonal degree matrix D (n x n),
  or NULL on error.
*/
MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A);

/*
Builds the normalized similarity matrix W.
Uses the formula: W = D^(-1/2) * A * D^(-1/2)

Input:
  A (MatrixPtr)- similarity matrix (n x n)
  D (MatrixPtr)- diagonal degree matrix (n x n)

Output:
  Returns the normalized similarity matrix W (n x n),
  or NULL on error.
*/
MatrixPtr getNormalizedSimilarityMatrix(MatrixPtr A, MatrixPtr D);

#ifndef PYTHON_BUILD

/*
Checks validity of program arguments.

Input:
  args_len - number of command line arguments
  goal     - operation type ("sym", "ddg", or "norm")

Output:
  Returns 1 if valid, 0 otherwise.
*/
int validity_check(int args_len, char *goal);

/*
Reads input data from a file, computes the requested matrix operation (sym, ddg, or norm), and prints the result.

Input (command line):
  argv[1] - goal string ("sym", "ddg", or "norm")
  argv[2] - path to input file

Output:
  Prints the requested matrix to stdout on success, "An Error Has Occurred" on error.
  Returns 0 on success, non-zero on error.
*/
int main(int argc, char *argv[]);

#endif

#endif