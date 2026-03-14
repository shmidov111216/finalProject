#include "matrix_util.h"

MatrixPtr create_matrix(int m, int n)
{
    MatrixPtr A = (MatrixPtr)calloc(1, sizeof(Matrix));
    if (!A)
    {
        printf("Failed to allocate Matrix struct");
        return NULL;
    }

    A->m = m;
    A->n = n;
    A->data = (double *)calloc(m, n*sizeof(double));
    if (!A->data)
    {
        printf("Failed to allocate Matrix data");
        free(A);
        return NULL;
    }

    return A;
}
// Free memory
int free_matrix(MatrixPtr A)
{
    CHECK_MATRIX_ALLOC(A);
    free(A->data);
    free(A);
    return SUCCESS;
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
        fprintf(stderr, "Index out of bounds get!\n");
        exit(EXIT_FAILURE);
    }
    return A->data[i * A->n + j];
}

// Set element
void mat_set(MatrixPtr A, int i, int j, double val)
{
    if (i < 0 || i >= A->m || j < 0 || j >= A->n)
    {
        fprintf(stderr, "Index out of bounds set!\n");
        exit(EXIT_FAILURE);
    }
    A->data[i * A->n + j] = val;
}

// Transpose (returns new matrix)
MatrixPtr mat_transpose(MatrixPtr A)
{
    MatrixPtr T = create_matrix(A->n, A->m);

    CHECK_MATRIX_ALLOC(T);

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

    CHECK_MATRIX_ALLOC(C);

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
        fprintf(stderr, "Error: dimensions do not match for element-wise multiplication1.\n");
        exit(EXIT_FAILURE);
    }
    MatrixPtr C = create_matrix(A->m, A->n);
    
    CHECK_MATRIX_ALLOC(C);

    for (int i = 0; i < A->m * A->n; i++)
        C->data[i] = A->data[i] * B->data[i];
    return C;
}

// Element-wise multiplication in-place A *= B
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dimensions do not match for element-wise multiplication2.\n");
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

// replace all zeroes with 1e-6
void replace_zeroes(MatrixPtr A)
{
    int size, i;

    if (A == NULL || A->data == NULL)
    {
        return;
    }

    size = A->m * A->n;

    for (i = 0; i < size; i++)
    {
        if (A->data[i] == 0.0)
        {
            A->data[i] = 1e-6;
        }
    }
}

// return the vector which is Row_i(X) - Row_j(X)
MatrixPtr get_row_diff(MatrixPtr X, int i, int j){
    int size = X->n;
    MatrixPtr diffVector = create_matrix(1, size);
    CHECK_MATRIX_ALLOC(diffVector);
    int k;

    for (k = 0; k<size; k++){
        diffVector->data[k] = mat_get(X, i, k) - mat_get(X, j, k);
    }
    return diffVector;
}
// sum all matrix values
double mat_sum(MatrixPtr A){
    double sum = 0;
    int i, size = A->m * A->n;
    for (i=0; i<size; i++){
        sum += A->data[i];
    }
    return sum;
}
// get vector represnting the row of A
MatrixPtr get_row_vector(MatrixPtr A, int row){
    int size = A->n, j;
    MatrixPtr row_vector = create_matrix(1, size);

    CHECK_MATRIX_ALLOC(row_vector);

    for (j=0; j<size; j++){
        row_vector->data[j] = mat_get(A, row, j);
    }
    return row_vector;
}

// given mat X, return vector v where vi = sum(Row_i(X))
// return collumn vector
MatrixPtr sum_axis_0(MatrixPtr A){
    int row = A->m;
    MatrixPtr sumVector = create_matrix(row, 1);

    CHECK_MATRIX_ALLOC(sumVector);

    MatrixPtr tempRow;
    int k;
    for (k = 0; k<row; k++){
        tempRow = get_row_vector(A, k);

        CHECK_MATRIX_ALLOC(tempRow);

        sumVector->data[k] = mat_sum(tempRow);
        free_matrix(tempRow);
    }
    return sumVector;
}
// power by d each aii in matrix A
void diagonal_power_inplace(MatrixPtr A, double power)
{
    int i;
    double val;
    for (i = 0; i < A->m && i < A->n; i++)
    {
        val = mat_get(A, i, i);
        val = pow(val, power);
        mat_set(A, i, i, val);
    }
}