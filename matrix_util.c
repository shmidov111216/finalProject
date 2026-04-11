#include "matrix_util.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* --- MEMORY POOL DEFINITIONS --- */
static MemoryPool mainPoolObj;
static MemoryPool tempPoolObj;

static MemoryPool *mainPool = &mainPoolObj;
static MemoryPool *tempPool = &tempPoolObj;

void pool_init(MemoryPool *pool)
{
    if (pool != NULL)
        pool->head = NULL;
}

/* Wrapper for easy use */
void pool_register_choice(int which_pool, void *ptr)
{
    MemoryPool *pool = (which_pool == MAIN_POOL) ? mainPool : tempPool;
    (void)pool_register(pool, ptr); /* explicitly ignore return to avoid warning */
}

/* Add existing pointer to pool */
void *pool_register(MemoryPool *pool, void *ptr)
{
    Allocation *node;

    if (ptr == NULL || pool == NULL)
        return ptr;

    node = (Allocation *)malloc(sizeof(Allocation));
    if (node == NULL)
    {
        free_matrix((MatrixPtr)ptr); /* Emergency cleanup */
        return NULL;
    }

    node->ptr = ptr;
    node->next = pool->head;
    pool->head = node;
    return ptr;
}

/* malloc on given pool */
void *pool_alloc(MemoryPool *pool, size_t size)
{
    return pool_register(pool, malloc(size));
}

/* calloc on given pool */
void *pool_calloc(MemoryPool *pool, size_t num, size_t size)
{
    return pool_register(pool, calloc(num, size));
}

/* Free all memory on given pool */
void pool_free_all(int which_pool)
{
    MemoryPool *pool = (which_pool == MAIN_POOL) ? mainPool : tempPool;
    Allocation *current;
    Allocation *temp;

    if (pool == NULL) return;

    current = pool->head;
    while (current != NULL)
    {
        temp = current;
        if (current->ptr != NULL)
            free_matrix((MatrixPtr)current->ptr);
        current = current->next;
        free(temp); /* Use standard free() for the node itself */
    }
    pool->head = NULL;
}

void destroy_pools()
{
    /* 1. Free all matrices and all nodes in both pools */
    pool_free_all(MAIN_POOL);
    pool_free_all(TEMP_POOL);

    /* 2. If you had any other global resources (like a file pointer
          or a large internal buffer), close them here. */

    /* 3. Nullify pointers to prevent accidental use-after-free */
    mainPool->head = NULL;
    tempPool->head = NULL;
}

/* Call before using pools */
void init_pools()
{
    pool_init(mainPool);
    pool_init(tempPool);
}
/* --- END MEMORY POOL --- */

/* Create a matrix of size m x n in the given pool */
MatrixPtr create_matrix(int m, int n, int which_pool)
{
    MemoryPool *pool = NULL;
    MatrixPtr A;
    size_t total_size;

    total_size = sizeof(Matrix) + (size_t)m * n * sizeof(double);

    if (which_pool == REGULAR_ALLOC)
        A = (MatrixPtr)calloc(1, total_size);
    else
    {
        pool = (which_pool == MAIN_POOL) ? mainPool : tempPool;
        A = (MatrixPtr)pool_calloc(pool, 1, total_size);
    }
    CHECK_MATRIX_ALLOC(A);

    A->m = m;
    A->n = n;

    /* Data lives immediately after the struct */
    A->data = (double *)(A + 1);

    return A;
}

/* Free memory ,No pool involved */
int free_matrix(MatrixPtr A)
{
    if (A == NULL)
        return FAIL;

    free(A);
    return SUCCESS;
}

/* Print matrix */
void print_matrix(MatrixPtr A)
{
    int i, j;
    for (i = 0; i < A->m; i++)
    {
        for (j = 0; j < A->n; j++)
        {
            if (j > 0)
                printf(",");
            printf("%.4f", MAT(A, i, j));
        }
        printf("\n");
    }
}

/* Get element from matrix */
double mat_get(MatrixPtr A, int i, int j)
{
    return MAT(A, i, j);
}

/* Set element */
void mat_set(MatrixPtr A, int i, int j, double val)
{
    MAT(A, i, j) = val;
}

/* Transpose (returns new matrix) */
MatrixPtr mat_transpose(MatrixPtr A, int which_pool)
{
    MatrixPtr T;
    int i, j;

    T = create_matrix(A->n, A->m, which_pool);
    CHECK_MATRIX_ALLOC(T);

    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            MAT(T, j, i) = MAT(A, i, j);

    return T;
}

/* In-place addition A = A + B */
void mat_add_inplace(MatrixPtr A, MatrixPtr B)
{
    size_t k;
    size_t size = (size_t)A->m * A->n;

    for (k = 0; k < size; k++)
        A->data[k] += B->data[k];
}

/* In-place aij=aij+x for a scalar x */
void mat_add_scalar_inplace(MatrixPtr A, double scalar)
{
    int i, j;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            MAT(A, i, j) += scalar;
}

/* In-place aij=aij*x for a scalar x */
void mat_scalar_mult_inplace(MatrixPtr A, double scalar)
{
    int i, j;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            MAT(A, i, j) *= scalar;
}

/* Dot product */
MatrixPtr mat_dot(MatrixPtr A, MatrixPtr B, int which_pool)
{
    MatrixPtr C;
    int i, j, k;

    C = create_matrix(A->m, B->n, which_pool);
    CHECK_MATRIX_ALLOC(C);

    for (i = 0; i < A->m; i++)
        for (j = 0; j < B->n; j++)
        {
            double sum = 0.0;
            for (k = 0; k < A->n; k++)
                sum += MAT(A, i, k) * MAT(B, k, j);
            MAT(C, i, j) = sum;
        }

    return C;
}

/* Element-wise multiplication (returns new matrix) */
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B, int which_pool)
{
    MatrixPtr C;
    int i, j;

    C = create_matrix(A->m, A->n, which_pool);
    CHECK_MATRIX_ALLOC(C);

    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            MAT(C, i, j) = MAT(A, i, j) * MAT(B, i, j);

    return C;
}

/* Element-wise multiplication in-place A *= B */
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B)
{
    int i, j;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            MAT(A, i, j) *= MAT(B, i, j);
}

/* In-place reciprocal */
void mat_reciprocal_inplace(MatrixPtr A)
{
    int i, j;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
        {
            MAT(A, i, j) = 1.0 / MAT(A, i, j);
        }
}

/* frobenius norm squared */
double mat_norm_sq(MatrixPtr A)
{
    double sum = 0.0;
    int i, j;

    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            sum += MAT(A, i, j) * MAT(A, i, j);

    return sum;
}

/* replace all zeroes with 1e-6 (to avoid division by 0) */
void replace_zeroes(MatrixPtr A)
{
    int i, j;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            if (MAT(A, i, j) == 0.0)
                MAT(A, i, j) = 1e-6;
}

/* return the vector (v = Row_i(X) - Row_j(X)) */
MatrixPtr get_row_diff(MatrixPtr X, int i, int j, int which_pool)
{
    MatrixPtr diffVector;
    int k;
    int size;

    size = X->n;
    diffVector = create_matrix(1, size, which_pool);
    CHECK_MATRIX_ALLOC(diffVector);

    for (k = 0; k < size; k++)
        MAT(diffVector, 0, k) = MAT(X, i, k) - MAT(X, j, k);

    return diffVector;
}

/* return sum of all matrix values */
double mat_sum(MatrixPtr A)
{
    double sum = 0.0;
    int i, j;

    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            sum += MAT(A, i, j);

    return sum;
}

/* returns new column vector where (vi = sum(Row_i(X))) */
MatrixPtr sum_axis_0(MatrixPtr A, int which_pool)
{
    MatrixPtr sumVector;
    int i, j;
    double row_sum;

    sumVector = create_matrix(A->m, 1, which_pool);
    CHECK_MATRIX_ALLOC(sumVector);

    for (i = 0; i < A->m; i++)
    {
        row_sum = 0.0;
        for (j = 0; j < A->n; j++)
            row_sum += MAT(A, i, j);
        MAT(sumVector, i, 0) = row_sum;
    }

    return sumVector;
}

/* (aii=aii^p) where p a scalar  */
void diagonal_power_inplace(MatrixPtr A, double power)
{
    int i;
    int limit = (A->m < A->n) ? A->m : A->n;

    for (i = 0; i < limit; i++)
        MAT(A, i, i) = pow(MAT(A, i, i), power);
}

/* (D * A) where A,D are square matrices (of the same size) and D is diagonal */
MatrixPtr mat_dot_diagonal_left(MatrixPtr D, MatrixPtr A, int which_pool)
{
    MatrixPtr C;
    int i, j;
    double d_ii;

    C = create_matrix(A->m, A->n, which_pool);
    CHECK_MATRIX_ALLOC(C);

    for (i = 0; i < A->m; i++)
    {
        d_ii = MAT(D, i, i);
        for (j = 0; j < A->n; j++)
            MAT(C, i, j) = d_ii * MAT(A, i, j);
    }

    return C;
}

/* (A * D) where A,D are square matrices (of the same size) and D is diagonal */
MatrixPtr mat_dot_diagonal_right(MatrixPtr A, MatrixPtr D, int which_pool)
{
    MatrixPtr C;
    int i, j;
    double d_jj;

    C = create_matrix(A->m, A->n, which_pool);
    CHECK_MATRIX_ALLOC(C);

    for (i = 0; i < A->m; i++)
    {
        for (j = 0; j < A->n; j++)
        {
            d_jj = MAT(D, j, j);
            MAT(C, i, j) = MAT(A, i, j) * d_jj;
        }
    }

    return C;
}