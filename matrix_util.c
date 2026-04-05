#include "matrix_util.h"

#ifdef PYTHON_BUILD
#pragma message "PYTHON_BUILD IS ACTIVE"
#define ERROR_PRINT() printf("An Error Has Occured C")
#else
#pragma message "PYTHON_BUILD IS NOT ACTIVE"
#define ERROR_PRINT()
#endif


/* --- MEMORY POOL DEFINITIONS --- */
/* 1. Define ACTUAL objects, not just pointers, to avoid segfaults */
static MemoryPool mainPoolObj;
static MemoryPool tempPoolObj;

/* These pointers now point to the real objects above */
static MemoryPool *mainPool = &mainPoolObj;
static MemoryPool *tempPool = &tempPoolObj;

void pool_init(MemoryPool *pool)
{
    if (pool)
        pool->head = NULL;
}

/* wrapper for easy use */
void pool_register_choice(int which_pool, void *ptr){
    MemoryPool *pool = which_pool == MAIN_POOL ? mainPool : tempPool;
    pool_register(pool, ptr);
}

/* 2. Standardized to take the pool pointer directly */
void *pool_register(MemoryPool *pool, void *ptr)
{
    if (!ptr || !pool)
        return ptr;

    Allocation *node = (Allocation *)malloc(sizeof(Allocation));
    if (!node)
    {
        free_matrix((MatrixPtr)ptr); // Emergency cleanup
        return NULL;
    }
    node->ptr = ptr;
    node->next = pool->head;
    pool->head = node;
    return ptr;
}

void *pool_alloc(MemoryPool *pool, size_t size)
{
    return pool_register(pool, malloc(size));
}

void *pool_calloc(MemoryPool *pool, size_t num, size_t size)
{
    return pool_register(pool, calloc(num, size));
}

void pool_free_all(int which_pool)
{
    MemoryPool *pool = (which_pool == MAIN_POOL) ? mainPool : tempPool;
    Allocation *current = pool->head;
    while (current != NULL)
    {
        Allocation *temp = current;
        if (current->ptr)
        {
            /* Cast the void* to MatrixPtr so free_matrix knows what to do */
            free_matrix((MatrixPtr)current->ptr);
        }
        current = current->next;
        free(temp); // USE STANDARD free() for the node itself!
    }
    pool->head = NULL;
}

void init_pools()
{
    pool_init(mainPool);
    pool_init(tempPool);
}
/* --- END MEMORY POOL --- */


MatrixPtr create_matrix(int m, int n, int which_pool)
{
    // MemoryPool *pool;
    // MatrixPtr A;
    // if (!which_pool)
    //     A = (MatrixPtr)calloc(1, sizeof(Matrix));
    // else
    // {
    //     pool = which_pool == MAIN_POOL ? mainPool : tempPool;
    //     A = (MatrixPtr)pool_calloc(pool, 1, sizeof(Matrix));
    // }
    
    // if (!A)
    // {
    //     printf("Failed to allocate Matrix struct\n");
    //     return NULL;
    // }

    // A->m = m;
    // A->n = n;

    // // Allocate array of row pointers
    // A->data = (double **)calloc(m, sizeof(double *));
    // if (!A->data)
    // {
    //     printf("Failed to allocate Matrix row pointers\n");
    //     free(A);
    //     return NULL;
    // }

    // // Allocate each row independently (prevents massive contiguous block failure)
    // for (int i = 0; i < m; i++)
    // {
    //     A->data[i] = (double *)calloc(n, sizeof(double));
    //     if (!A->data[i])
    //     {
    //         // If allocation fails halfway, free previously allocated rows to prevent leak
    //         printf("Failed to allocate Matrix row %d\n", i);
    //         for (int j = 0; j < i; j++)
    //         {
    //             free(A->data[j]);
    //         }
    //         free(A->data);
    //         free(A);
    //         return NULL;
    //     }
    // }

    // return A;



    MemoryPool *pool = NULL;
    MatrixPtr A;

    size_t total_size = sizeof(Matrix) + (size_t)m * n * sizeof(double);

    if (which_pool == REGULAR_ALLOC)
    {
        A = (MatrixPtr)calloc(1, total_size);
    }
    else
    {
        pool = (which_pool == MAIN_POOL) ? mainPool : tempPool;
        A = (MatrixPtr)pool_calloc(pool, 1, total_size);
    }

    if (!A)
        return NULL;

    A->m = m;
    A->n = n;

    /* data lives immediately after the struct */
    A->data = (double *)(A + 1);

    return A;
}

// Free memory
int free_matrix(MatrixPtr A)
{
    // if (!A)
    //     return FAIL;

    // // Free each row first
    // for (int i = 0; i < A->m; i++)
    // {
    //     free(A->data[i]);
    // }
    // // Free the array of pointers, then the struct
    // free(A->data);
    // free(A);
    // return SUCCESS;

    if (!A)
        return FAIL;

    free(A);
    return SUCCESS;
}

// Print matrix
void print_matrix(MatrixPtr A)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
            printf("%.4f ", MAT(A, i, j));
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
    return MAT(A, i, j);
}

// Set element
void mat_set(MatrixPtr A, int i, int j, double val)
{
    if (i < 0 || i >= A->m || j < 0 || j >= A->n)
    {
        fprintf(stderr, "Index out of bounds set!\n");
        exit(EXIT_FAILURE);
    }
    MAT(A, i, j) = val;
}

// Transpose (returns new matrix)
MatrixPtr mat_transpose(MatrixPtr A, int which_pool)
{
    MatrixPtr T = create_matrix(A->n, A->m, which_pool);
    CHECK_MATRIX_ALLOC(T);

    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            MAT(T, j, i) = MAT(A, i, j);
    return T;
}

// In-place addition A += B
void mat_add_inplace(MatrixPtr A, MatrixPtr B)
{
    if (A->m != B->m || A->n != B->n)
        exit(EXIT_FAILURE);

    size_t size = (size_t)A->m * A->n;
    for (size_t k = 0; k < size; k++)
        A->data[k] += B->data[k];
}

void mat_add_scalar_inplace(MatrixPtr A, double scalar)
{
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            MAT(A, i, j) += scalar;
}

void mat_scalar_mult_inplace(MatrixPtr A, double scalar)
{
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            MAT(A, i, j) *= scalar;
}

// Dot product
MatrixPtr mat_dot(MatrixPtr A, MatrixPtr B, int which_pool)
{
    if (A->n != B->m)
        exit(EXIT_FAILURE);

    MatrixPtr C = create_matrix(A->m, B->n, which_pool);
    CHECK_MATRIX_ALLOC(C);

    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < B->n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < A->n; k++)
                sum += MAT(A,i,k) * MAT(B,k,j);

            MAT(C,i,j) = sum;
        }
    }
    return C;
}

// Element-wise multiplication (returns new matrix)
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B, int which_pool)
{
    if (A->m != B->m || A->n != B->n)
    {
        fprintf(stderr, "Error: dim mismatch for element-wise mult1.\n");
        exit(EXIT_FAILURE);
    }
    MatrixPtr C = create_matrix(A->m, A->n, which_pool);
    CHECK_MATRIX_ALLOC(C);

    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            MAT(C, i, j) = MAT(A, i, j) * MAT(B, i, j);
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
            MAT(A, i, j) *= MAT(B, i, j);
}

// In-place reciprocal
void mat_reciprocal_inplace(MatrixPtr A)
{
    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
        {
            if (MAT(A, i, j) == 0.0)
            {
                fprintf(stderr, "Error: div by zero at [%d][%d].\n", i, j);
                exit(EXIT_FAILURE);
            }
            MAT(A, i, j) = 1.0 / MAT(A, i, j);
        }
    }
}

// frobenius norm squared
double mat_norm_sq(MatrixPtr A)
{
    double sum = 0.0;
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            sum += MAT(A, i, j) * MAT(A, i, j);
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
            if (MAT(A, i, j) == 0.0)
                MAT(A, i, j) = 1e-6;
        }
    }
}

// return the vector which is Row_i(X) - Row_j(X)
MatrixPtr get_row_diff(MatrixPtr X, int i, int j, int which_pool)
{
    int size = X->n;
    MatrixPtr diffVector = create_matrix(1, size, which_pool);
    CHECK_MATRIX_ALLOC(diffVector);

    for (int k = 0; k < size; k++)
        MAT(diffVector, 0, k) = MAT(X, i, k) - MAT(X, j, k);

    return diffVector;
}
// sum all matrix values
double mat_sum(MatrixPtr A)
{
    double sum = 0;
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            sum += MAT(A, i, j);
    return sum;
}

// get vector representing the row of A
/*
MatrixPtr get_row_vector(MatrixPtr A, int row, int which_pool)
{
    int size = A->n;
    MatrixPtr row_vector = create_matrix(1, size, which_pool);
    CHECK_MATRIX_ALLOC(row_vector);

    for (int j = 0; j < size; j++)
        row_vector->data[0][j] = A->data[row][j];

    return row_vector;
}
*/

// return column vector where vi = sum(Row_i(X))
MatrixPtr sum_axis_0(MatrixPtr A, int which_pool)
{
    MatrixPtr sumVector = create_matrix(A->m, 1, which_pool);
    CHECK_MATRIX_ALLOC(sumVector);

    for (int i = 0; i < A->m; i++)
    {
        double row_sum = 0.0;
        for (int j = 0; j < A->n; j++)
            row_sum += MAT(A, i, j);

        MAT(sumVector, i, 0) = row_sum;
    }
    return sumVector;
}

// power by d each aii in matrix A
void diagonal_power_inplace(MatrixPtr A, double power)
{
    int limit = (A->m < A->n) ? A->m : A->n;
    for (int i = 0; i < limit; i++)
        MAT(A, i, i) = pow(MAT(A, i, i), power);
}

// Left Dot Diagonal D * A
MatrixPtr mat_dot_diagonal_left(MatrixPtr D, MatrixPtr A, int which_pool)
{
    if (D->m != A->m)
    {
        fprintf(stderr, "Error: D * A left multiply dim mismatch.\n");
        exit(EXIT_FAILURE);
    }

    MatrixPtr C = create_matrix(A->m, A->n, which_pool);
    if (!C)
        return NULL;

    for (int i = 0; i < A->m; i++)
    {
        double d_ii = MAT(D, i, i);
        for (int j = 0; j < A->n; j++)
        {
            MAT(C, i, j) = d_ii * MAT(A, i, j);
        }
    }
    return C;
}

// Right Dot Diagonal A * D
MatrixPtr mat_dot_diagonal_right(MatrixPtr A, MatrixPtr D, int which_pool)
{
    if (A->n != D->m)
    {
        fprintf(stderr, "Error: A * D right multiply dim mismatch.\n");
        exit(1);
    }

    MatrixPtr C = create_matrix(A->m, A->n, which_pool);
    if (!C)
        return NULL;

    for (int i = 0; i < A->m; i++)
    {
        for (int j = 0; j < A->n; j++)
        {
            double d_jj = MAT(D, j, j);
            MAT(C, i, j) = MAT(A, i, j) * d_jj;
        }
    }
    return C;
}