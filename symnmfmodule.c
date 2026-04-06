#include "symnmf.h"

#define PY_SSIZE_T_CLEAN
#define MODULE
#include <Python.h>
#include "matrix_util.h"

/* ========= FORWARD DECLARATIONS ========= */

MatrixPtr convertPyObjToMatrix(PyObject *obj, int which_pool);
PyObject *matrix_to_pylist(MatrixPtr mat, int k, int d);
static PyObject *symnmf(PyObject *self, PyObject *args);

/* ======================================== */

/* -------- Conversion Utilities ---------- */
MatrixPtr convertPyObjToMatrix(PyObject *obj, int which_pool)
{
    int N;
    int d;
    int i, j;
    PyObject *first_row;
    PyObject *row;
    MatrixPtr mat;

    N = PyList_GET_SIZE(obj);
    first_row = PyList_GET_ITEM(obj, 0);
    d = PyList_GET_SIZE(first_row);

    mat = create_matrix(N, d, which_pool);
    CHECK_MATRIX_ALLOC(mat);

    for (i = 0; i < N; i++)
    {
        row = PyList_GET_ITEM(obj, i);
        for (j = 0; j < d; j++)
        {
            MAT(mat, i, j) = PyFloat_AsDouble(
                PyList_GET_ITEM(row, j));
        }
    }
    return mat;
}

PyObject *matrix_to_pylist(MatrixPtr mat, int k, int d)
{
    int i, j;
    PyObject *outer;
    PyObject *row;
    PyObject *val;

    outer = PyList_New(k);
    if (!outer)
        return NULL;

    for (i = 0; i < k; i++)
    {
        row = PyList_New(d);
        if (!row)
            return NULL;

        for (j = 0; j < d; j++)
        {
            val = PyFloat_FromDouble(mat_get(mat, i, j));
            if (!val)
                return NULL;

            PyList_SetItem(row, j, val);
        }

        PyList_SetItem(outer, i, row);
    }

    return outer;
}

/* -------- Python Wrapper ---------- */

static PyObject *symnmf(PyObject *self, PyObject *args)
{
    PyObject *H_py;
    PyObject *W_py;
    PyObject *res_H_py;

    int n;
    int k;

    MatrixPtr H;
    MatrixPtr W;
    MatrixPtr res_H;

    (void)self;

    init_pools();

    if (!PyArg_ParseTuple(args, "OOii", &H_py, &W_py, &n, &k))
        return NULL;

    H = convertPyObjToMatrix(H_py, REGULAR_ALLOC);
    CHECK_FREE_AND_EXIT(H);

    W = convertPyObjToMatrix(W_py, MAIN_POOL);
    CHECK_FREE_AND_EXIT(W);

    res_H = getResultH(H, W);
    CHECK_FREE_AND_EXIT(res_H);

    res_H_py = matrix_to_pylist(res_H, n, k);
    CHECK_FREE_AND_EXIT(res_H_py);


    pool_free_all(MAIN_POOL);
    pool_free_all(TEMP_POOL);

    return res_H_py;
}

static PyObject *sym(PyObject *self, PyObject *args)
{
    PyObject *X_py;
    PyObject *A_py;

    int n;
    int d;

    MatrixPtr X;
    MatrixPtr A;

    (void)self;

    init_pools();

    if (!PyArg_ParseTuple(args, "Oii", &X_py, &n, &d))
        return NULL;

    X = convertPyObjToMatrix(X_py, MAIN_POOL);
    CHECK_FREE_AND_EXIT(X);

    A = getSimilarityMatrix(X);
    CHECK_FREE_AND_EXIT(A);

    A_py = matrix_to_pylist(A, n, n);
    CHECK_FREE_AND_EXIT(A_py);

    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);

    return A_py;
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    PyObject *A_py;
    PyObject *D_py;

    int n;

    MatrixPtr A;
    MatrixPtr D;

    (void)self;

    init_pools();

    if (!PyArg_ParseTuple(args, "Oi", &A_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py, MAIN_POOL);
    CHECK_FREE_AND_EXIT(A);

    D = getDiagonalDegreeMatrix(A);
    CHECK_FREE_AND_EXIT(D);

    D_py = matrix_to_pylist(D, n, n);
    CHECK_FREE_AND_EXIT(D_py);

    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);

    return D_py;
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    PyObject *W_py;
    PyObject *A_py;
    PyObject *D_py;

    int n;

    MatrixPtr W;
    MatrixPtr A;
    MatrixPtr D;

    (void)self;

    init_pools();

    if (!PyArg_ParseTuple(args, "OOi", &A_py, &D_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py, MAIN_POOL);
    CHECK_FREE_AND_EXIT(A);

    D = convertPyObjToMatrix(D_py, MAIN_POOL);
    CHECK_FREE_AND_EXIT(D);

    W = getNormalizedSimilarityMatrix(A, D);
    CHECK_FREE_AND_EXIT(W);

    W_py = matrix_to_pylist(W, n, n);
    CHECK_FREE_AND_EXIT(W_py);

    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);

    return W_py;
}

/* -------- Module Definition ---------- */

static PyMethodDef symnmfMethods[] = {
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS,
     "Returns the optimized solution H for clustering.\n"
     "Args:\n"
     "  H (list of list of float): initial matrix.\n"
     "  W (list of list of float): normalized similarity matrix.\n"
     "  n (int): number of points.\n"
     "  k (int): number of clusters."},

    {"sym", (PyCFunction)sym, METH_VARARGS,
     "Returns the similarity matrix A.\n"
     "Args:\n"
     "  X (list of list of float): data points.\n"
     "  n (int): number of points.\n"
     "  d (int): dimension."},

    {"ddg", (PyCFunction)ddg, METH_VARARGS,
     "Returns the diagonal degree matrix D.\n"
     "Args:\n"
     "  A (list of list of float): similarity matrix.\n"
     "  n (int): number of points."},

    {"norm", (PyCFunction)norm, METH_VARARGS,
     "Returns the normalized similarity matrix W.\n"
     "Args:\n"
     "  A (list of list of float): similarity matrix.\n"
     "  D (list of list of float): diagonal degree matrix.\n"
     "  n (int): number of points."},

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