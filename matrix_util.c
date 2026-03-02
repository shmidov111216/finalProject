#include "matrix_util.h"

// Create a new matrix
MatrixPtr create_matrix(int m, int n)
{
    MatrixPtr A = (MatrixPtr)malloc(sizeof(Matrix));
    if (!A)
    {
        fprintf(stderr, "Error: malloc failed for Matrix struct.\n");
        exit(EXIT_FAILURE);
    }
    A->m = m;
    A->n = n;
    A->data = (double *)calloc(m * n, sizeof(double));
    if (!A->data)
    {
        fprintf(stderr, "Error: calloc failed for Matrix data.\n");
        free(A);
        exit(EXIT_FAILURE);
    }
    return A;
}

// Free memory
void free_matrix(MatrixPtr A)
{
    if (!A)
        return;
    free(A->data);
    free(A);
}

// Print matrix
void print_matrix(MatrixPtr A)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
            printf("%8.3f ", A->data[i * A->n + j]);
        printf("\n");
    }
}

// Get element
double mat_get(MatrixPtr A, int i, int j)
{
    if (i < 0 || i >= A->m || j < 0 || j >= A->n)
    {
        fprintf(stderr, "Index out of bounds!\n");
        exit(EXIT_FAILURE);
    }
    return A->data[i * A->n + j];
}

// Set element
void mat_set(MatrixPtr A, int i, int j, double val)
{
    if (i < 0 || i >= A->m || j < 0 || j >= A->n)
    {
        fprintf(stderr, "Index out of bounds!\n");
        exit(EXIT_FAILURE);
    }
    A->data[i * A->n + j] = val;
}

// Transpose (returns new matrix)
MatrixPtr mat_transpose(MatrixPtr A)
{
    MatrixPtr T = create_matrix(A->n, A->m);
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            T->data[j * T->n + i] = A->data[i * A->n + j];
    return T;
}

// In-place addition A += B
void mat_add_inplace(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dimensions do not match for addition.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->m * A->n; i++)
        A->data[i] += B->data[i];
}

void mat_add_scalar_inplace(MatrixPtr A, double scalar)
{
    for (int i = 0; i < A->m * A->n; i++)
        A->data[i] += scalar;
}

void mat_scalar_mult_inplace(MatrixPtr A, double scalar)
{
    for (int i = 0; i < A->m * A->n; i++)
        A->data[i] *= scalar;
}

// Dot product (matrix multiplication, returns new matrix)
MatrixPtr mat_dot(MatrixPtr A, MatrixPtr B)
{
    if (A->n != B->m)
    {
        fprintf(stderr, "Error: incompatible dimensions for dot product.\n");
        exit(EXIT_FAILURE);
    }
    MatrixPtr C = create_matrix(A->m, B->n);
    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < B->n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < A->n; k++)
                sum += A->data[i * A->n + k] * B->data[k * B->n + j];
            C->data[i * C->n + j] = sum;
        }
    }
    return C;
}

// Element-wise multiplication (returns new matrix)
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dimensions do not match for element-wise multiplication.\n");
        exit(EXIT_FAILURE);
    }
    MatrixPtr C = create_matrix(A->m, A->n);
    for (int i = 0; i < A->m * A->n; i++)
        C->data[i] = A->data[i] * B->data[i];
    return C;
}

// Element-wise multiplication in-place A *= B
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dimensions do not match for element-wise multiplication.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->m * A->n; i++)
        A->data[i] *= B->data[i];
}

// In-place reciprocal
void mat_reciprocal_inplace(MatrixPtr A)
{
    for (int i = 0; i < A->m * A->n; i++)
    {
        if (A->data[i] == 0.0)
        {
            fprintf(stderr, "Error: division by zero in mat_reciprocal_inplace at index %d.\n", i);
            exit(EXIT_FAILURE);
        }
        A->data[i] = 1.0 / A->data[i];
    }
}

// frobinius norm squared
double mat_norm_sq(MatrixPtr A)
{
    double sum = 0.0;
    int size = A->m * A->n;

    for (int i = 0; i < size; i++)
        sum += A->data[i] * A->data[i];

    return sum;
}