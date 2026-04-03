#include "matrix_util.h"

MatrixPtr create_matrix(int m, int n)
{
    MatrixPtr A = (MatrixPtr)calloc(1, sizeof(Matrix));
    if (!A)
    {
        printf("Failed to allocate Matrix struct\n");
        return NULL;
    }

    A->m = m;
    A->n = n;

    // Allocate array of row pointers
    A->data = (double **)calloc(m, sizeof(double *));
    if (!A->data)
    {
        printf("Failed to allocate Matrix row pointers\n");
        free(A);
        return NULL;
    }

    // Allocate each row independently (prevents massive contiguous block failure)
    for (int i = 0; i < m; i++)
    {
        A->data[i] = (double *)calloc(n, sizeof(double));
        if (!A->data[i])
        {
            // If allocation fails halfway, free previously allocated rows to prevent leak
            printf("Failed to allocate Matrix row %d\n", i);
            for (int j = 0; j < i; j++)
            {
                free(A->data[j]);
            }
            free(A->data);
            free(A);
            return NULL;
        }
    }

    return A;
}

// Free memory
int free_matrix(MatrixPtr A)
{
    if (!A)
        return FAIL;

    // Free each row first
    for (int i = 0; i < A->m; i++)
    {
        free(A->data[i]);
    }
    // Free the array of pointers, then the struct
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
            printf("%.4f ", A->data[i][j]);
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
    return A->data[i][j];
}

// Set element
void mat_set(MatrixPtr A, int i, int j, double val)
{
    if (i < 0 || i >= A->m || j < 0 || j >= A->n)
    {
        fprintf(stderr, "Index out of bounds set!\n");
        exit(EXIT_FAILURE);
    }
    A->data[i][j] = val;
}

// Transpose (returns new matrix)
MatrixPtr mat_transpose(MatrixPtr A)
{
    MatrixPtr T = create_matrix(A->n, A->m);
    CHECK_MATRIX_ALLOC(T);

    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            T->data[j][i] = A->data[i][j];
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
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            A->data[i][j] += B->data[i][j];
}

void mat_add_scalar_inplace(MatrixPtr A, double scalar)
{
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            A->data[i][j] += scalar;
}

void mat_scalar_mult_inplace(MatrixPtr A, double scalar)
{
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            A->data[i][j] *= scalar;
}

// Dot product
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
                sum += A->data[i][k] * B->data[k][j];
            C->data[i][j] = sum;
        }
    }
    return C;
}

// Element-wise multiplication (returns new matrix)
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dim mismatch for element-wise mult1.\n");
        exit(EXIT_FAILURE);
    }
    MatrixPtr C = create_matrix(A->m, A->n);
    CHECK_MATRIX_ALLOC(C);

    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            C->data[i][j] = A->data[i][j] * B->data[i][j];
    return C;
}

// Element-wise multiplication in-place A *= B
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dim mismatch for element-wise mult2.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            A->data[i][j] *= B->data[i][j];
}

// In-place reciprocal
void mat_reciprocal_inplace(MatrixPtr A)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
        {
            if (A->data[i][j] == 0.0)
            {
                fprintf(stderr, "Error: div by zero at [%d][%d].\n", i, j);
                exit(EXIT_FAILURE);
            }
            A->data[i][j] = 1.0 / A->data[i][j];
        }
    }
}

// frobenius norm squared
double mat_norm_sq(MatrixPtr A)
{
    double sum = 0.0;
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            sum += A->data[i][j] * A->data[i][j];
    return sum;
}

// replace all zeroes with 1e-6
void replace_zeroes(MatrixPtr A)
{
    if (A == NULL || A->data == NULL)
        return;

    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
        {
            if (A->data[i][j] == 0.0)
                A->data[i][j] = 1e-6;
        }
    }
}

// return the vector which is Row_i(X) - Row_j(X)
MatrixPtr get_row_diff(MatrixPtr X, int i, int j)
{
    int size = X->n;
    MatrixPtr diffVector = create_matrix(1, size);
    CHECK_MATRIX_ALLOC(diffVector);

    for (int k = 0; k < size; k++)
        diffVector->data[0][k] = X->data[i][k] - X->data[j][k];

    return diffVector;
}

// sum all matrix values
double mat_sum(MatrixPtr A)
{
    double sum = 0;
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            sum += A->data[i][j];
    return sum;
}

// get vector representing the row of A
MatrixPtr get_row_vector(MatrixPtr A, int row)
{
    int size = A->n;
    MatrixPtr row_vector = create_matrix(1, size);
    CHECK_MATRIX_ALLOC(row_vector);

    for (int j = 0; j < size; j++)
        row_vector->data[0][j] = A->data[row][j];

    return row_vector;
}

// return column vector where vi = sum(Row_i(X))
MatrixPtr sum_axis_0(MatrixPtr A)
{
    MatrixPtr sumVector = create_matrix(A->m, 1);
    CHECK_MATRIX_ALLOC(sumVector);

    for (int i = 0; i < A->m; i++)
    {
        double row_sum = 0;
        for (int j = 0; j < A->n; j++)
            row_sum += A->data[i][j];

        sumVector->data[i][0] = row_sum;
    }
    return sumVector;
}

// power by d each aii in matrix A
void diagonal_power_inplace(MatrixPtr A, double power)
{
    for (int i = 0; i < A->m && i < A->n; i++)
    {
        A->data[i][i] = pow(A->data[i][i], power);
    }
}

// Left Dot Diagonal D * A
MatrixPtr mat_dot_diagonal_left(MatrixPtr D, MatrixPtr A)
{
    if (D->m != A->m)
    {
        fprintf(stderr, "Error: D * A left multiply dim mismatch.\n");
        exit(EXIT_FAILURE);
    }

    MatrixPtr C = create_matrix(A->m, A->n);
    if (!C)
        return NULL;

    for (int i = 0; i < A->m; i++)
    {
        double d_ii = D->data[i][i];
        for (int j = 0; j < A->n; j++)
        {
            C->data[i][j] = d_ii * A->data[i][j];
        }
    }
    return C;
}

// Right Dot Diagonal A * D
MatrixPtr mat_dot_diagonal_right(MatrixPtr A, MatrixPtr D)
{
    if (A->n != D->m)
    {
        fprintf(stderr, "Error: A * D right multiply dim mismatch.\n");
        exit(1);
    }

    MatrixPtr C = create_matrix(A->m, A->n);
    if (!C)
        return NULL;

    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
        {
            double d_jj = D->data[j][j];
            C->data[i][j] = A->data[i][j] * d_jj;
        }
    }
    return C;
}