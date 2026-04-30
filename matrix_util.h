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
#define ERROR_PRINT() PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred")
#define ERROR_CODE ((void *)(0))
#else
#define ERROR_PRINT() printf("An Error Has Occurred\n")
#define ERROR_CODE 1
#endif

#define REGULAR_ALLOC 0
#define MAIN_POOL 1
#define TEMP_POOL 2

#define MAT(A,i,j) ((A)->data[(size_t)(i) * (A)->n + (j)])

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


/* --- MEMORY POOL FUNCTIONS --- */

/*
Initializes a memory pool.

Sets the pool to an empty state.

Input:
  pool - memory pool to initialize

Output:
  None
*/
void pool_init(MemoryPool *pool);

/*
Registers a pointer in the selected memory pool.

Input:
  which_pool - selects the target pool (e.g. MAIN_POOL or TEMP_POOL)
  ptr        - pointer to register

Output:
  None.
*/
void pool_register_choice(int which_pool, void *ptr);

/*
Registers an allocated pointer in a memory pool for later cleanup.

If registration fails, the pointer is freed immediately.

Input:
  pool - memory pool to register in
  ptr  - allocated pointer to track

Output:
  Returns the pointer on success,
  returns NULL if registration fails.
*/
void *pool_register(MemoryPool *pool, void *ptr);

/*
Allocates memory and registers it in a memory pool.

Input:
  pool - memory pool to track the allocation
  size - number of bytes to allocate

Output:
  Returns pointer to allocated memory on success,
  or NULL if allocation fails.
*/
void *pool_alloc(MemoryPool *pool, size_t size);

/*
Allocates zero-initialized memory and registers it in a memory pool.

Input:
  pool - memory pool to track the allocation
  num  - number of elements
  size - size of each element

Output:
  Returns pointer to allocated memory on success,
  or NULL if allocation fails.
*/
void *pool_calloc(MemoryPool *pool, size_t num, size_t size);

/*
Frees all allocations stored in a selected memory pool.

Each tracked pointer is freed, then all tracking nodes are released.
The pool is left empty.

Input:
  which_pool - selects the pool to clear (MAIN_POOL or TEMP_POOL)

Output:
  None.
*/
void pool_free_all(int which_pool);

/*
Destroys all memory pools and frees all tracked allocations.

Frees all memory in both pools and resets them to an empty state.
Also clears internal pool pointers to prevent reuse.

Input:
  None.

Output:
  None.
*/
void destroy_pools();

/*
Initializes all global memory pools.

Sets each pool to an empty state.

Input:
  None

Output:
  None
*/
void init_pools();
/* --- END MEMORY POOL --- */


/* --- MATRIX CORE OPERATIONS --- */

/*
Creates a matrix with the given number of rows and columns.
The matrix is allocated either normally or through a memory pool.

Input:
  m         - number of rows
  n         - number of columns
  which_pool - allocation method or pool to use

Output:
  Returns a pointer to the new matrix, or NULL on allocation failure.
*/
MatrixPtr create_matrix(int m, int n, int which_pool);

/*
Frees a matrix.

Input:
  A - matrix to free

Output:
  Returns SUCCESS on success, FAIL if A is NULL.
*/
int free_matrix(MatrixPtr A);

/*
Prints a matrix in CSV-style format.

Input:
  A - matrix to print

Output:
  None.
*/
void print_matrix(MatrixPtr A);

/*
Returns the value at a given matrix position.

Input:
  A - matrix to read from
  i - row index
  j - column index

Output:
  Returns the value at position (i, j), or FAIL if A is NULL.
*/
double mat_get(MatrixPtr A, int i, int j);

/*
Sets the value at a given matrix position.

Input:
  A   - matrix to modify
  i   - row index
  j   - column index
  val - value to store

Output:
  None.
*/
void mat_set(MatrixPtr A, int i, int j, double val);
/* --- END MATRIX CORE OPERATIONS --- */


/* --- MATRIX NUMERICAL OPERATIONS AND TRANSFORMATIONS --- */

/*
Creates and returns the transpose of a matrix.

Input:
  A          - matrix to transpose
  which_pool - allocation method or pool to use

Output:
  Returns the transposed matrix,
  or NULL on error.
*/
MatrixPtr mat_transpose(MatrixPtr A, int which_pool);

/*
Adds matrix B to matrix A in place.

Input:
  A - matrix to modify
  B - matrix to add

Output:
  None.
*/
void mat_add_inplace(MatrixPtr A, MatrixPtr B);

/*
Adds a scalar value to all entries of a matrix in place.

Input:
  A      - matrix to modify
  scalar - value to add

Output:
  None.
*/
void mat_add_scalar_inplace(MatrixPtr A, double scalar);

/*
Multiplies all entries of a matrix by a scalar in place.

Input:
  A      - matrix to modify
  scalar - multiplication factor

Output:
  None.
*/
void mat_scalar_mult_inplace(MatrixPtr A, double scalar);

/*
Computes the matrix product C = A * B.

Input:
  A          - left matrix
  B          - right matrix
  which_pool - allocation method or pool to use

Output:
  Returns the product matrix,
  or NULL on error.
*/
MatrixPtr mat_dot(MatrixPtr A, MatrixPtr B, int which_pool);

/*
Computes element-wise multiplication of two matrices.

Input:
  A          - first matrix
  B          - second matrix
  which_pool - allocation method or pool to use

Output:
  Returns a new matrix with element-wise products,
  or NULL on error.
*/
MatrixPtr mat_elementwise_prod(MatrixPtr A, MatrixPtr B, int which_pool);

/*
Performs element-wise multiplication in place (A *= B).

Input:
  A - matrix to modify
  B - matrix to multiply with

Output:
  None.
*/
void mat_elementwise_prod_inplace(MatrixPtr A, MatrixPtr B);

/*
Replaces each entry with its reciprocal (1 / value) in place.

Input:
  A - matrix to modify

Output:
  None.
*/
void mat_reciprocal_inplace(MatrixPtr A);

/*
Computes the squared Frobenius norm of a matrix.

Input:
  A - matrix

Output:
  Returns the squared norm,
  or FAIL if A is NULL.
*/
double mat_norm_sq(MatrixPtr A);

/*
Adds a small constant (1e-6) to every matrix entry in place. (To avoid division by 0)

Input:
  A - matrix to modify

Output:
  None.
*/
void add_infinitesimal_inplace(MatrixPtr A);

/*
Computes the difference between two rows of a matrix.

Input:
  X          - source matrix
  i          - first row index
  j          - second row index
  which_pool - allocation method or pool to use

Output:
  Returns a row vector equal to Row_i(X) - Row_j(X),
  or NULL on error.
*/
MatrixPtr get_row_diff(MatrixPtr X, int i, int j, int which_pool);

/*
Computes the sum of all matrix elements.

Input:
  A - matrix

Output:
  Returns the sum of all values,
  or -1 if A is NULL.
*/
double mat_sum(MatrixPtr A);

/*
Computes the sum of each row and returns it as a column vector.

Input:
  A          - source matrix
  which_pool - allocation method or pool to use

Output:
  Returns a column vector of row sums,
  or NULL on error.
*/
MatrixPtr sum_axis_0(MatrixPtr A, int which_pool);

/*
Raises each diagonal entry of a matrix to a given power in place.

Input:
  A     - matrix to modify
  power - exponent to apply

Output:
  None.
*/
void diagonal_power_inplace(MatrixPtr A, double power);

/*
Computes the product C = D * A, where D is a diagonal matrix.

Input:
  D          - diagonal matrix
  A          - matrix to multiply
  which_pool - allocation method or pool to use

Output:
  Returns the resulting matrix,
  or NULL on error.
*/
MatrixPtr mat_dot_diagonal_left(MatrixPtr D, MatrixPtr A, int which_pool);

/*
Computes the product C = A * D, where D is a diagonal matrix.

Input:
  A          - matrix to multiply
  D          - diagonal matrix
  which_pool - allocation method or pool to use

Output:
  Returns the resulting matrix,
  or NULL on error.
*/
MatrixPtr mat_dot_diagonal_right(MatrixPtr A, MatrixPtr D, int which_pool);
/* --- END MATRIX NUMERICAL OPERATIONS AND TRANSFORMATIONS --- */

#endif