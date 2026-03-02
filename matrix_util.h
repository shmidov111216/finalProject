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

// Pointer typedef
typedef Matrix *MatrixPtr;

// Function prototypes
MatrixPtr create_matrix(int m, int n); // allocate matrix
void free_matrix(MatrixPtr A);         // free memory
void print_matrix(MatrixPtr A);        // print
double mat_get(MatrixPtr A, int i, int j);
void mat_set(MatrixPtr A, int i, int j, double val);

// Operations
MatrixPtr mat_transpose(MatrixPtr A);                     // returns new matrix
void mat_add_inplace(MatrixPtr A, MatrixPtr B);           // A += B
void mat_scalar_mult_inplace(MatrixPtr A, double scalar); // inplace
MatrixPtr mat_dot(MatrixPtr A, MatrixPtr B);              // returns new matrix
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B); // returns new matrix
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B);
void mat_reciprocal_inplace(MatrixPtr A);
void mat_add_scalar_inplace(MatrixPtr A, double scalar);
double mat_norm_sq(MatrixPtr A); // squared norm

#endif