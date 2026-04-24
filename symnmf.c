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

    /* following given equation in assignment*/
    W_H = mat_dot(W, H, TEMP_POOL);
    CHECK_MATRIX_ALLOC(W_H);

    Ht = mat_transpose(H, TEMP_POOL);
    CHECK_MATRIX_ALLOC(Ht);

    H_Ht = mat_dot(H, Ht, TEMP_POOL);
    CHECK_MATRIX_ALLOC(H_Ht);

    H_Ht_H = mat_dot(H_Ht, H, TEMP_POOL);
    CHECK_MATRIX_ALLOC(H_Ht_H);

    /* fill H_Ht_H zeroes to avoid deviding by zero */
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

    /* setting H = H_updated - H to check convergence */
    /* note -- H is being overridden (it is no longer needed) */
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
                CHECK_MATRIX_ALLOC(diffVector);

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
    CHECK_MATRIX_ALLOC(tmp);
    
    W = mat_dot_diagonal_right(tmp, D, MAIN_POOL);
    /* Don't need CHECK_MATRIX_ALLOC(W) since we check it in main()*/

    pool_free_all(TEMP_POOL);
    return W;
}

/* Do not compile for python module */
#ifndef PYTHON_BUILD

int main(int argc, char *argv[]){
    char *file_name, *goal;
    FILE *file;
    MatrixPtr X, A, D, W;
    if (argc < 3)
        return 1;

    goal = argv[1];
    file_name = argv[2];

    file = fopen(file_name, "r");
    if (file == NULL) return 1;

    init_pools();
    X = parse_matrix_from_stream(file);
    if(!X) goto check_free_and_exit;

    A = getSimilarityMatrix(X);
    if(!A) goto check_free_and_exit;
    if (strcmp("sym", goal) == 0) /* print similarity matrix if goal="sym" */
        print_matrix(A);
    
    D = getDiagonalDegreeMatrix(A);
    if(!D) goto check_free_and_exit;
    if (strcmp("ddg", goal) == 0) /* print diagonal degree matrix if goal="ddg" */
        print_matrix(D);

    W = getNormalizedSimilarityMatrix(A, D);
    if(!W) goto check_free_and_exit;
    if (strcmp("norm", goal) == 0) /* print normalized similarity matrix if goal="norm" */
        print_matrix(W);

    fclose(file);
    destroy_pools();
    return 0;

    check_free_and_exit:
        ERROR_PRINT();
        fclose(file);
        destroy_pools();
        return ERROR_CODE;
}

#endif