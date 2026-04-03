#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H
#define SUCCESS 1
#define FAIL 0
#define CHECK_MATRIX_ALLOC(x)   \
    do                   \
    {                    \
        if (!(x))        \
            return FAIL; \
    } while (0)


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
int free_matrix(MatrixPtr A);         // free memory
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
void replace_zeroes(MatrixPtr A);
double mat_norm_sq(MatrixPtr A); // squared norm
MatrixPtr get_row_diff(MatrixPtr X, int i, int j);
double mat_sum(MatrixPtr A);
MatrixPtr sum_axis_0(MatrixPtr A);
void diagonal_power_inplace(MatrixPtr A, double power);
MatrixPtr mat_dot_diagonal_left(MatrixPtr D, MatrixPtr A);
MatrixPtr mat_dot_diagonal_right(MatrixPtr A, MatrixPtr D);

#endif