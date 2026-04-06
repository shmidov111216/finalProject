#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define FAIL 0
#define SUCCESS 1

#define IN_PYTHON 0
#define IN_C 1

#ifdef PYTHON_BUILD
#define ERROR_PRINT() ((void)0)
#define ERROR_CODE ((void *)(0))
#else
#define ERROR_PRINT() printf("An Error Has Occurred C\n")
#define ERROR_CODE 1
#endif

#define REGULAR_ALLOC 0
#define MAIN_POOL 1
#define TEMP_POOL 2

#define MAT(A,i,j) ((A)->data[(size_t)(i) * (A)->n + (j)])

#define CHECK_MATRIX_ALLOC(ptr) \
    do                          \
    {                           \
        if (!(ptr))             \
            return NULL;        \
    } while (0)

#define CHECK_FREE_AND_EXIT(ptr)      \
    do                                \
    {                                 \
        if (!(ptr))                   \
        {                             \
            ERROR_PRINT();            \
            pool_free_all(MAIN_POOL); \
            pool_free_all(TEMP_POOL); \
            return ERROR_CODE;        \
        }                             \
    } while (0)

/* Matrix struct */
typedef struct
{
    int m;          /* rows */
    int n;          /* columns */
    double *data;   /* contiguous m*n block */
} Matrix;

typedef struct Allocation
{
    void *ptr;
    struct Allocation *next;
} Allocation;

typedef struct
{
    Allocation *head;
} MemoryPool;

/* Pointer typedef */
typedef Matrix *MatrixPtr;

/* Function prototypes */
MatrixPtr create_matrix(int m, int n, int which_pool);
int free_matrix(MatrixPtr A);
void print_matrix(MatrixPtr A);
double mat_get(MatrixPtr A, int i, int j);
void mat_set(MatrixPtr A, int i, int j, double val);

/* Operations */
MatrixPtr mat_transpose(MatrixPtr A, int which_pool);
void mat_add_inplace(MatrixPtr A, MatrixPtr B);
void mat_scalar_mult_inplace(MatrixPtr A, double scalar);
MatrixPtr mat_dot(MatrixPtr A, MatrixPtr B, int which_pool);
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B, int which_pool);
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B);
void mat_reciprocal_inplace(MatrixPtr A);
void mat_add_scalar_inplace(MatrixPtr A, double scalar);
void replace_zeroes(MatrixPtr A);
double mat_norm_sq(MatrixPtr A);
MatrixPtr get_row_diff(MatrixPtr X, int i, int j, int which_pool);
double mat_sum(MatrixPtr A);
MatrixPtr sum_axis_0(MatrixPtr A, int which_pool);
void diagonal_power_inplace(MatrixPtr A, double power);
MatrixPtr mat_dot_diagonal_left(MatrixPtr D, MatrixPtr A, int which_pool);
MatrixPtr mat_dot_diagonal_right(MatrixPtr A, MatrixPtr D, int which_pool);

/* Pool functions */
void pool_init(MemoryPool *pool);
void *pool_register(MemoryPool *pool, void *ptr);
void *pool_alloc(MemoryPool *pool, size_t size);
void *pool_calloc(MemoryPool *pool, size_t num, size_t size);
void *pool_realloc(MemoryPool *pool, void *old_ptr, size_t new_size);
void pool_free_all(int which_pool);
void init_pools();
void pool_register_choice(int which_pool, void *ptr);

#endif