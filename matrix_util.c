// matrix_util.c
#include "matrix_util.h"

// Create a new matrix with all entries initialized to 0
Matrix create_matrix(int m, int n)
{
    Matrix A;
    A.m = m;
    A.n = n;
    A.data = (double *)calloc(m * n, sizeof(double));
    return A;
}

// Free memory
void free_matrix(Matrix A)
{
    free(A.data);
}

// Print matrix
void print_matrix(Matrix A)
{
    for (int i = 0; i < A.m; i++)
    {
        for (int j = 0; j < A.n; j++)
        {
            printf("%8.3f ", A.data[i * A.n + j]);
        }
        printf("\n");
    }
}

// Transpose
Matrix transpose(Matrix A)
{
    Matrix T = create_matrix(A.n, A.m);
    for (int i = 0; i < A.m; i++)
        for (int j = 0; j < A.n; j++)
            T.data[j * T.n + i] = A.data[i * A.n + j];
    return T;
}

// Addition
Matrix add(Matrix A, Matrix B)
{
    if (A.m != B.m || A.n != B.n)
    {
        fprintf(stderr, "Error: dimensions do not match for addition.\n");
        exit(1);
    }
    Matrix C = create_matrix(A.m, A.n);
    for (int i = 0; i < A.m * A.n; i++)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

// Scalar multiplication
Matrix scalar_mult(Matrix A, double scalar)
{
    Matrix B = create_matrix(A.m, A.n);
    for (int i = 0; i < A.m * A.n; i++)
        B.data[i] = A.data[i] * scalar;
    return B;
}

// Dot product (matrix multiplication)
Matrix dot(Matrix A, Matrix B)
{
    if (A.n != B.m)
    {
        fprintf(stderr, "Error: incompatible dimensions for dot product.\n");
        exit(1);
    }
    Matrix C = create_matrix(A.m, B.n);
    for (int i = 0; i < A.m; i++)
    {
        for (int j = 0; j < B.n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < A.n; k++)
                sum += A.data[i * A.n + k] * B.data[k * B.n + j];
            C.data[i * C.n + j] = sum;
        }
    }
    return C;
}
