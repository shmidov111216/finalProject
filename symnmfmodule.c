#include <stdio.h>
#include <stdlib.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "matrix_util.h"
/* --- MEMORY POOL DEFINITIONS --- */

typedef struct Allocation
{
    void *ptr;
    struct Allocation *next;
} Allocation;

typedef struct
{
    Allocation *head;
} MemoryPool;

/* Initialize the pool */
void pool_init(MemoryPool *pool)
{
    pool->head = NULL;
}

/* Registers an existing pointer into the pool */
void *pool_register(MemoryPool *pool, void *ptr)
{
    Allocation *node;
    if (!ptr)
        return NULL;

    node = (Allocation *)malloc(sizeof(Allocation));
    if (!node)
    {
        free(ptr);
        return NULL;
    }
    node->ptr = ptr;
    node->next = pool->head;
    pool->head = node;
    return ptr;
}

/* Wrappers for standard allocation functions */
void *pool_alloc(MemoryPool *pool, size_t size)
{
    void *ptr = malloc(size);
    return pool_register(pool, ptr);
}

void *pool_calloc(MemoryPool *pool, size_t num, size_t size)
{

    void *ptr = calloc(num, size);
    return pool_register(pool, ptr);
}

void *pool_realloc(MemoryPool *pool, void *old_ptr, size_t new_size)
{
    Allocation *node;
    void *new_ptr;

    if (old_ptr == NULL)
        return pool_alloc(pool, new_size);

    /* Search for the existing pointer in our registry */
    node = pool->head;
    while (node != NULL)
    {
        if (node->ptr == old_ptr)
        {
            new_ptr = realloc(old_ptr, new_size);
            if (!new_ptr)
                return NULL; /* Realloc failed, old_ptr still valid */

            node->ptr = new_ptr; /* Update our records to the new address */
            return new_ptr;
        }
        node = node->next;
    }
    return NULL; /* Trying to realloc something not in the pool */
}

/* Frees everything in the list */
void pool_free_all(MemoryPool *pool)
{
    Allocation *current = pool->head;
    Allocation *temp;
    while (current != NULL)
    {
        temp = current;
        if (current->ptr)
            free(current->ptr);
        current = current->next;
        free(temp);
    }
    pool->head = NULL;
}
/* --- END MEMORY POOL --- */

/* Custom string functions (to avoid string.h dependency) */
size_t my_strlen(const char *s)
{
    size_t i = 0;
    while (s[i] != '\0')
        i++;
    return i;
}

void *my_memcpy(void *dest, const void *src, size_t n)
{
    size_t i;
    char *d = (char *)dest;
    const char *s = (const char *)src;
    for (i = 0; i < n; i++)
        d[i] = s[i];
    return dest;
}

char *my_strcpy(char *dest, const char *src)
{
    char *p = dest;
    while ((*p++ = *src++) != '\0')
        ;
    return dest;
}

MatrixPtr convertPyObjToMatrix(PyObject *obj)
{
    int N, d;
    double val;
    MatrixPtr mat;
    if (!PyList_Check(obj))
        return NULL;

    N = PyList_Size(obj);
    d = PyList_Size(PyList_GetItem(obj, 0));

    mat = create_matrix(N, d);
    if (!mat)
    {
        return NULL;
    }

    for (int i = 0; i < N; i++)
    {
        PyObject *row = PyList_GetItem(obj, i);

        for (int j = 0; j < d; j++)
        {
            val = PyFloat_AsDouble(PyList_GetItem(row, j));
            mat_set(mat, i, j, val);
        }
    }

    return mat;
}

PyObject *matrix_to_pylist(MatrixPtr mat, int k, int d)
{
    int i, j;
    PyObject *outer = PyList_New(k);
    if (!outer)
        return NULL;

    for (i = 0; i < k; i++)
    {
        PyObject *row = PyList_New(d);
        if (!row)
            return NULL;

        for (j = 0; j < d; j++)
        {
            PyObject *val = PyFloat_FromDouble(mat_get(mat, i, j));
            if (!val)
                return NULL;

            PyList_SetItem(row, j, val);
        }

        PyList_SetItem(outer, i, row);
    }
    return outer;
}

static PyObject *fit(PyObject *self, PyObject *args)
{
    PyObject H_py, W_py, res_H_py;
    int n, k;
    MatrixPtr H, W, res_H;

    /* This parses the Python arguments into a double (d)  variable named z and int (i) variable named n*/
    if (!PyArg_ParseTuple(args, "OOiii", H_py, W_py, n, k))
    {
        return NULL; /* In the CPython API, a NULL value is never valid for a
                        PyObject* so it is used to signal that an error has occurred. */
    }
    H = convertPyObjToMatrix(H_py);
    W = convertPyObjToMatrix(W_py);

    if (!H || !W)
    {
        if (H)
            free_matrix(H);
        if (W)
            free_matrix(W);
        return NULL;
    }

    res_H = getResultH(H, W);
    res_H_py = matrix_to_pylist(res_H, n, k);

    free_matrix(res_H)

    return res_H_py;
}

MatrixPtr updateH(MatrixPtr H, MatrixPtr W)
{
    double beta = 0.5;
    MatrixPtr H_updated, W_H, Ht, H_Ht, H_Ht_H;
    W_H = mat_dot(W, H);
    Ht = mat_transpose(H);
    H_Ht = mat_dot(H, Ht);
    H_Ht_H = mat_dot(Ht, H);

    // make every entry 1/x
    mat_reciprocal_inplace(H_Ht_H);
    mat_elementwise_prod_inplace(W_H, H_Ht_H);
    mat_scalar_mult_inplace(W_H, beta);
    mat_add_scalar_inplace(W_H, 1 - beta);

    H_updated = mat_elementwise_prod(H, W_H);

    free_matrix(W_H);
    free_matrix(Ht);
    free_matrix(H_Ht);
    free_matrix(H_Ht_H);
    
    return H_updated;
}

int checkConvergence(MatrixPtr H, MatrixPtr H_updated, double epsilon) {
    int isConverged;
    mat_scalar_mult_inplace(H, -1);
    mat_add_inplace(H, H_updated);
    isConverged = mat_norm_sq(H) < epsilon;
    return isConverged;
}

// run at most maxIter updates to initial H
MatrixPtr getResultH(MatrixPtr H, MatrixPtr W) {
    const double epsilon = 1e-4;
    const int maxIter = 300;
    int t;
    MatrixPtr H_updated;
    for (t = 0; t < maxIter; t++){
        H_updated = updateH(H, W);
        // converge
        if (checkConvergence(H, H_updated, epsilon)){
            free(H);
            return H_updated;
        }
        free(H);
        H = H_updated;
    }
    return H;
}

static PyMethodDef kmeansmethods[] = {
    {"fit",            /* the Python method name that will be used */
     (PyCFunction)fit, /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,     /* flags indicating parameters
accepted for this function */
     PyDoc_STR("fit runs kmeans given:\n"
               "MatrixPtr H,\n"
               "MatrixPtr W,\n"
               "int k,\n"
               "int n,\n")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}    /* The last entry must be all NULL as shown to act as a
                                sentinel. Python looks for this entry to know that all
                                of the functions for the module have been defined. */
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf", /* name of module */
    NULL,         /* module documentation, may be NULL */
    -1,           /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    kmeansmethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_symnmf(void)
{
    PyObject *m;
    m = PyModule_Create(&kmeansmodule);
    if (!m)
    {
        return NULL;
    }
    return m;
}
