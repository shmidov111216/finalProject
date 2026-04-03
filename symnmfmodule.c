
#include "symnmf.h"
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "matrix_util.h"

/* ========= FORWARD DECLARATIONS ========= */

MatrixPtr convertPyObjToMatrix(PyObject *obj);
PyObject *matrix_to_pylist(MatrixPtr mat, int k, int d);
static PyObject *symnmf(PyObject *self, PyObject *args);

/* ======================================== */

/* -------- Conversion Utilities ---------- */
MatrixPtr convertPyObjToMatrix(PyObject *obj)
{
    int N = PyList_GET_SIZE(obj);
    PyObject *first_row = PyList_GET_ITEM(obj, 0);
    int d = PyList_GET_SIZE(first_row);
    init_pools();

    MatrixPtr mat = create_matrix(N, d, MAIN_POOL);
    CHECK_MATRIX_ALLOC(mat);

    for (int i = 0; i < N; i++)
    {
        PyObject *row = PyList_GET_ITEM(obj, i);
        double *target_row = mat->data[i]; // Get the row pointer once

        for (int j = 0; j < d; j++)
        {
            // PyList_GET_ITEM is a macro (no function call overhead)
            // PyFloat_AsDouble is necessary to extract the raw 8-byte double
            target_row[j] = PyFloat_AsDouble(PyList_GET_ITEM(row, j));
        }
    }
    return mat;
}

PyObject *matrix_to_pylist(MatrixPtr mat, int k, int d)
{
    int i, j;
    PyObject *outer = PyList_New(k);
    init_pools();

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

            PyList_SetItem(row, j, val); // steals reference
        }

        PyList_SetItem(outer, i, row); // steals reference
    }

    return outer;
}



/* -------- Python Wrapper ---------- */

static PyObject *symnmf(PyObject *self, PyObject *args)
{
    printf("hello from symnmf module!\n");
    PyObject *H_py, *W_py, *res_H_py;
    int n, k;
    MatrixPtr H, W, res_H;
    init_pools();

    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "OOii", &H_py, &W_py, &n, &k))
        return NULL;

    H = convertPyObjToMatrix(H_py);
    printf("H created\n");
    W = convertPyObjToMatrix(W_py);
    
    printf("W created\n");
    

    printf("created matrices success!\n");
    exit(0);

    if (!H || !W)
    {
        FREE_AND_EXIT();
    }

    printf("before algorithm\n");
    
    res_H = getResultH(H, W);
    if (!H)
    {
        FREE_AND_EXIT();
    }

    res_H_py = matrix_to_pylist(res_H, n, k);
    if (!res_H_py)
    {
        FREE_AND_EXIT();
    }
    
    printf("after algorithm\n");

    pool_free_all(MAIN_POOL);
    pool_free_all(TEMP_POOL);
    return res_H_py;
}

static PyObject *sym(PyObject *self, PyObject *args)
{
    PyObject *X_py, *A_py;
    int n, d;
    MatrixPtr X, A;

    init_pools();

    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "Oii", &X_py, &n, &d))
        return NULL;

    X = convertPyObjToMatrix(X_py);

    if (!X)
    {
        FREE_AND_EXIT();
    }
    // todo
    A = getSimilarityMatrix(X);

    if (!A)
    {
        FREE_AND_EXIT();
    }

    A_py = matrix_to_pylist(A, n, n);
    
    if (!A_py)
    {
        FREE_AND_EXIT();
    }

    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);
    return A_py;
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    PyObject *A_py, *D_py;
    int n;
    MatrixPtr A, D;
    init_pools();

    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "Oi", &A_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py);

    if (!A)
    {
        FREE_AND_EXIT();
    }

    D = getDiagonalDegreeMatrix(A);

    if (!D){
        FREE_AND_EXIT();
    }

    D_py = matrix_to_pylist(D, n, n);

    if (!D_py){
        FREE_AND_EXIT();
    }

    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);
    
    return D_py;
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    PyObject *W_py, *A_py, *D_py;
    int n;
    MatrixPtr W, A, D;
    init_pools();
    
    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "OOi", &A_py, &D_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py);
    D = convertPyObjToMatrix(D_py);

    if (!D || !A){
        FREE_AND_EXIT();
    }
    W = getNormalizedSimilarityMatrix(A, D);

    if (!W){
        FREE_AND_EXIT();
    }

    W_py = matrix_to_pylist(W, n, n);
    
    if (!W_py){
        FREE_AND_EXIT();
    }

    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);
    
    return W_py;
}

/* -------- Module Definition ---------- */

static PyMethodDef symnmfMethods[] = {
    {"symnmf",
     (PyCFunction)symnmf,
     METH_VARARGS,
     PyDoc_STR("Runs SymNMF")},
    {"sym",
     (PyCFunction)sym,
     METH_VARARGS,
     PyDoc_STR("Runs sym")},
    {"ddg",
     (PyCFunction)ddg,
     METH_VARARGS,
     PyDoc_STR("Runs ddg")},
    {"norm",
     (PyCFunction)norm,
     METH_VARARGS,
     PyDoc_STR("Runs norm")},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    symnmfMethods};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}