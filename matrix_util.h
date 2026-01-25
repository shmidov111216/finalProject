// matrix_util.h
#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

#include <stdlib.h>
#include <stdio.h>

// Matrix struct
typedef struct
{
    int m;        // rows
    int n;        // columns
    double *data; // flattened array, row-major
} Matrix;

// Function prototypes
Matrix create_matrix(int m, int n); // allocate matrix
void free_matrix(Matrix A);         // free memory
void print_matrix(Matrix A);        // print

Matrix transpose(Matrix A);
Matrix add(Matrix A, Matrix B);
Matrix scalar_mult(Matrix A, double scalar);
Matrix dot(Matrix A, Matrix B);

#endif
