#include "symnmf.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* -------- Core Algorithm ---------- */

MatrixPtr updateH(MatrixPtr H, MatrixPtr W)
{
    double beta = 0.5;
    MatrixPtr H_updated, W_H, Ht, H_Ht, H_Ht_H;

    W_H = mat_dot(W, H, TEMP_POOL);
    CHECK_MATRIX_ALLOC(W_H);

    Ht = mat_transpose(H, TEMP_POOL);
    CHECK_MATRIX_ALLOC(Ht);

    H_Ht = mat_dot(H, Ht, TEMP_POOL);
    CHECK_MATRIX_ALLOC(H_Ht);

    H_Ht_H = mat_dot(H_Ht, H, TEMP_POOL);
    CHECK_MATRIX_ALLOC(H_Ht_H);

    /* fill H_Ht_H zeroes */
    replace_zeroes(H_Ht_H);
    mat_reciprocal_inplace(H_Ht_H);
    mat_elementwise_prod_inplace(W_H, H_Ht_H);
    mat_scalar_mult_inplace(W_H, beta);
    mat_add_scalar_inplace(W_H, 1 - beta);

    H_updated = mat_elementwise_prod(H, W_H, REGULAR_ALLOC);
    CHECK_MATRIX_ALLOC(H_updated);

    pool_free_all(TEMP_POOL);

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
        CHECK_MATRIX_ALLOC(H_updated);

        if (checkConvergence(H, H_updated, epsilon))
        {
            free_matrix(H);
            pool_register_choice(MAIN_POOL, H_updated);
            return H_updated;
        }

        free_matrix(H);
        H = H_updated;
    }

    pool_register_choice(MAIN_POOL, H);
    return H;
}

MatrixPtr getSimilarityMatrix(MatrixPtr X)
{
    MatrixPtr A, diffVector;
    int i, j;
    double val;

    A = create_matrix(X->m, X->m, MAIN_POOL);
    CHECK_MATRIX_ALLOC(A);

    for (i = 0; i < X->m; i++)
    {
        for (j = 0; j < X->m; j++)
        {
            if (i == j)
            {
                mat_set(A, i, j, 0);
            }
            else
            {
                diffVector = get_row_diff(X, i, j, TEMP_POOL);
                val = exp(-0.5 * mat_norm_sq(diffVector));
                mat_set(A, i, j, val);
            }
        }
    }

    pool_free_all(TEMP_POOL);
    return A;
}

MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A)
{
    int i;
    double value;
    MatrixPtr D, sumVector;

    D = create_matrix(A->m, A->m, MAIN_POOL);
    CHECK_MATRIX_ALLOC(D);

    /* sumVector is a col vector */
    sumVector = sum_axis_0(A, TEMP_POOL);
    CHECK_MATRIX_ALLOC(sumVector);

    for (i = 0; i < D->m; i++)
    {
        value = mat_get(sumVector, i, 0);
        mat_set(D, i, i, value);
    }

    pool_free_all(TEMP_POOL);
    return D;
}

MatrixPtr getNormalizedSimilarityMatrix(MatrixPtr A, MatrixPtr D)
{
    MatrixPtr W, tmp;

    diagonal_power_inplace(D, -0.5);

    tmp = mat_dot_diagonal_left(D, A, TEMP_POOL);
    W = mat_dot_diagonal_right(tmp, D, MAIN_POOL);

    pool_free_all(TEMP_POOL);
    return W;
}
MatrixPtr parse_matrix_from_stream(FILE *fp)
{
    size_t line_cap;
    char *line;

    int rows;
    int cols;
    int total_cap;
    int total_used;
    int bufferCnt;

    double *buffer;

    MatrixPtr X;

    /* init */
    line_cap = 1024;
    line = (char *)malloc(line_cap);
    CHECK_MATRIX_ALLOC(line);

    rows = 0;
    cols = -1;
    bufferCnt = 0;

    total_cap = INIT_CAP;
    total_used = 0;

    buffer = (double *)malloc(total_cap * sizeof(double));
    CHECK_MATRIX_ALLOC(buffer);

    while (fgets(line, (int)line_cap, fp))
    {
        size_t len;

        len = strlen(line);

        /* grow line buffer if needed (no limit on line length) */
        while (len == line_cap - 1 && line[len - 1] != '\n')
        {
            char *tmp_line;

            line_cap *= 2;
            tmp_line = (char *)realloc(line, line_cap);
            CHECK_MATRIX_ALLOC(tmp_line);
            line = tmp_line;

            if (!fgets(line + len, (int)(line_cap - len), fp))
                break;

            len = strlen(line);
        }

        /* tokenize */
        {
            char *token;
            int col_count;

            token = strtok(line, ", \t\n");
            col_count = 0;

            while (token)
            {
                if (total_used >= total_cap)
                {
                    double *tmp_buf;

                    total_cap *= 2;
                    tmp_buf = (double *)realloc(buffer, total_cap * sizeof(double));
                    CHECK_MATRIX_ALLOC(tmp_buf);
                    buffer = tmp_buf;
                }

                buffer[total_used] = atof(token);
                total_used++;
                col_count++;

                token = strtok(NULL, ", \t\n");
            }

            if (col_count == 0)
                continue; /* skip empty lines */

            if (cols == -1)
                cols = col_count;
            else if (cols != col_count)
            {
                free(buffer);
                free(line);
                return NULL;
            }

            rows++;
        }
    }

    free(line);

    if (rows == 0 || cols <= 0)
    {
        free(buffer);
        return NULL;
    }

    X = create_matrix(rows, cols, MAIN_POOL);
    CHECK_MATRIX_ALLOC(X);

    {
        int i;
        int j;

        bufferCnt = 0;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                MAT(X, i, j) = buffer[bufferCnt];
                bufferCnt++;
            }
        }
    }
    X = X;
    free(buffer);
    return X;
}

#ifndef PYTHON_BUILD
int main(int argc, char *argv[])
{
    char *file_name;
    char *goal;
    FILE *file;
    MatrixPtr X, A, D, W;

    if (argc < 3)
        return 1;

    goal = argv[1];
    file_name = argv[2];

    file = fopen(file_name, "r");
    if (file == NULL)
        return 1;

    init_pools();

    X = parse_matrix_from_stream(file);
    CHECK_FREE_AND_EXIT(X);

    A = getSimilarityMatrix(X);
    CHECK_FREE_AND_EXIT(A);

    if (strcmp("sym", goal) == 0)
        print_matrix(A);

    D = getDiagonalDegreeMatrix(A);
    CHECK_FREE_AND_EXIT(D);

    if (strcmp("ddg", goal) == 0)
        print_matrix(D);

    W = getNormalizedSimilarityMatrix(A, D);
    CHECK_FREE_AND_EXIT(W);

    if (strcmp("norm", goal) == 0)
        print_matrix(W);

    fclose(file);
    pool_free_all(TEMP_POOL);
    pool_free_all(MAIN_POOL);
    return 0;
}

#endif