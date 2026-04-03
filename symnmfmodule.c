
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
    if (!PyList_Check(obj))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    int N = PyList_Size(obj);
    if (N == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Input list is empty");
        return NULL;
    }

    PyObject *first_row = PyList_GetItem(obj, 0);
    if (!PyList_Check(first_row))
    {
        PyErr_SetString(PyExc_TypeError, "Each row must be a list");
        return NULL;
    }

    int d = PyList_Size(first_row);
    if (d == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Rows cannot be empty");
        return NULL;
    }

    MatrixPtr mat = create_matrix(N, d);
    if (!mat)
        return NULL; // Exception already set

    for (int i = 0; i < N; i++)
    {
        PyObject *row = PyList_GetItem(obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != d)
        {
            free_matrix(mat);
            PyErr_SetString(PyExc_TypeError, "All rows must have the same length");
            return NULL;
        }

        for (int j = 0; j < d; j++)
        {
            double val = PyFloat_AsDouble(PyList_GetItem(row, j));
            if (PyErr_Occurred())
            { // conversion failed
                free_matrix(mat);
                return NULL;
            }
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
    
    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "OOii", &H_py, &W_py, &n, &k))
        return NULL;

    H = convertPyObjToMatrix(H_py);
    W = convertPyObjToMatrix(W_py);
    printf("created matrices success!\n");

    if (!H || !W)
    {
        if (H)
            free_matrix(H);
        if (W)
            free_matrix(W);
        return NULL;
    }

    printf("before algorithm\n");
    res_H = getResultH(H, W);
    res_H_py = matrix_to_pylist(res_H, n, k);
    printf("after algorithm\n");
    free_matrix(res_H);
    return res_H_py;
}

static PyObject *sym(PyObject *self, PyObject *args)
{
    PyObject *X_py, *A_py;
    int n, d;
    MatrixPtr X, A;

    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "Oii", &X_py, &n, &d))
        return NULL;

    X = convertPyObjToMatrix(X_py);

    if (!X)
    {
        return NULL;
    }
    // todo
    A = getSimilarityMatrix(X);
    A_py = matrix_to_pylist(A, n, n);

    free_matrix(A);
    free_matrix(X);
    return A_py;
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    PyObject *A_py, *D_py;
    int n;
    MatrixPtr A, D;

    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "Oi", &A_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py);

    if (!A)
    {
        return NULL;
    }

    D = getDiagonalDegreeMatrix(A);
    D_py = matrix_to_pylist(D, n, n);

    free_matrix(D);
    return D_py;
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    PyObject *W_py, *A_py, *D_py;
    int n;
    MatrixPtr W, A, D;

    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "OOi", &A_py, &D_py, &n))
        return NULL;

    A = convertPyObjToMatrix(A_py);
    D = convertPyObjToMatrix(D_py);

    if (!D || !A)
    {
        if (A)
            free_matrix(A);
        if (D)
            free_matrix(D);
        return NULL;
    }
    W = getNormalizedSimilarityMatrix(A, D);
    // TODO maybe free all if function failed
    W_py = matrix_to_pylist(W, n, n);
    free_matrix(W);
    free_matrix(A);
    free_matrix(D);
    // TODO maybe label and W_py=null
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