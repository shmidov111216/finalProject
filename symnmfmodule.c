#include <stdio.h>
#include <stdlib.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "matrix_util.h"

/* ========= FORWARD DECLARATIONS ========= */

MatrixPtr convertPyObjToMatrix(PyObject *obj);
PyObject *matrix_to_pylist(MatrixPtr mat, int k, int d);

MatrixPtr updateH(MatrixPtr H, MatrixPtr W);
int checkConvergence(MatrixPtr H, MatrixPtr H_updated, double epsilon);
MatrixPtr getResultH(MatrixPtr H, MatrixPtr W);

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

/* -------- Core Algorithm ---------- */

MatrixPtr updateH(MatrixPtr H, MatrixPtr W)
{
    double beta = 0.5;
    MatrixPtr H_updated, W_H, Ht, H_Ht, H_Ht_H;

    W_H = mat_dot(W, H);
    Ht = mat_transpose(H);
    H_Ht = mat_dot(H, Ht);
    H_Ht_H = mat_dot(H_Ht, H);

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

int checkConvergence(MatrixPtr H, MatrixPtr H_updated, double epsilon)
{
    int isConverged;

    mat_scalar_mult_inplace(H, -1);
    mat_add_inplace(H, H_updated);
    isConverged = mat_norm_sq(H) < epsilon;

    return isConverged;
}

MatrixPtr getResultH(MatrixPtr H, MatrixPtr W)
{
    const double epsilon = 1e-4;
    const int maxIter = 300;
    int t;
    MatrixPtr H_updated;
    for (t = 0; t < maxIter; t++)
    {
        H_updated = updateH(H, W);

        if (checkConvergence(H, H_updated, epsilon))
        {
            free_matrix(H);
            return H_updated;
        }

        free_matrix(H);
        H = H_updated;
    }

    return H;
}

/* -------- Python Wrapper ---------- */

static PyObject *symnmf(PyObject *self, PyObject *args)
{
    PyObject *H_py, *W_py, *res_H_py;
    int n, k;
    MatrixPtr H, W, res_H;
    
    /* Parse arguments correctly */
    if (!PyArg_ParseTuple(args, "OOii", &H_py, &W_py, &n, &k))
        return NULL;

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

    free_matrix(res_H);
    return res_H_py;
}

/* -------- Module Definition ---------- */

static PyMethodDef symnmfMethods[] = {
    {"symnmf",
     (PyCFunction)symnmf,
     METH_VARARGS,
     PyDoc_STR("Runs SymNMF")},
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