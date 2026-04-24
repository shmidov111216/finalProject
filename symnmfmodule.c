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
    int N,d,i,j;
    PyObject *first_row, *row;
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
    PyObject *outer, *row, *val;
    int i, j;


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
    PyObject *H_py, *W_py, *res_H_py;
    MatrixPtr H, W, res_H;
    int n, k;
    
    init_pools();

    if (!PyArg_ParseTuple(args, "OOii", &H_py, &W_py, &n, &k))
        return NULL;

    H = convertPyObjToMatrix(H_py, REGULAR_ALLOC);
    if (!H) goto check_free_and_exit;

    W = convertPyObjToMatrix(W_py, MAIN_POOL);
    if (!W) goto check_free_and_exit;

    res_H = getResultH(H, W);
    if (!res_H) goto check_free_and_exit;

    res_H_py = matrix_to_pylist(res_H, n, k);
    if (!res_H_py) goto check_free_and_exit;


    destroy_pools();

    return res_H_py;
    
    check_free_and_exit:
        ERROR_PRINT();
        destroy_pools();
        return ERROR_CODE;
}

static PyObject *sym(PyObject *self, PyObject *args)
{
    PyObject *X_py, *A_py;
    MatrixPtr X, A;
    int n, d;

    init_pools();

    if (!PyArg_ParseTuple(args, "Oii", &X_py, &n, &d))
        return NULL;

    X = convertPyObjToMatrix(X_py, MAIN_POOL);
    if (!X) goto check_free_and_exit;

    A = getSimilarityMatrix(X);
    if (!A) goto check_free_and_exit;

    A_py = matrix_to_pylist(A, n, n);
    if (!A_py) goto check_free_and_exit;

    destroy_pools();
    return A_py;

    check_free_and_exit:
        ERROR_PRINT();
        destroy_pools();
        return ERROR_CODE;
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    PyObject *A_py, *D_py;
    MatrixPtr A, D;
    int n;
    init_pools();

    if (!PyArg_ParseTuple(args, "Oi", &A_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py, MAIN_POOL);
    if (!A) goto check_free_and_exit;

    D = getDiagonalDegreeMatrix(A);
    if (!D) goto check_free_and_exit;

    D_py = matrix_to_pylist(D, n, n);
    if (!D_py) goto check_free_and_exit;

    destroy_pools();

    return D_py;

    check_free_and_exit:
        ERROR_PRINT();
        destroy_pools();
        return ERROR_CODE;
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    PyObject *W_py, *A_py, *D_py;
    MatrixPtr W, A, D;
    int n;
    
    init_pools();

    if (!PyArg_ParseTuple(args, "OOi", &A_py, &D_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py, MAIN_POOL);
    if(!A) goto check_free_and_exit;

    D = convertPyObjToMatrix(D_py, MAIN_POOL);
    if (!D) goto check_free_and_exit;

    W = getNormalizedSimilarityMatrix(A, D);
    if (!W) goto check_free_and_exit;

    W_py = matrix_to_pylist(W, n, n);
    if(!W_py) goto check_free_and_exit;

    destroy_pools();

    return W_py;

    check_free_and_exit:
        ERROR_PRINT();
        destroy_pools();
        return ERROR_CODE;
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

PyMODINIT_FUNC PyInit_symnmfmodule(void){
    return PyModule_Create(&symnmfmodule);
}