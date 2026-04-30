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
    /* check allocation success in each step to avoid segfault*/
    W_H = mat_dot(W, H, TEMP_POOL);
    Ht = mat_transpose(H, TEMP_POOL);
    H_Ht = mat_dot(H, Ht, TEMP_POOL);
    H_Ht_H = mat_dot(H_Ht, H, TEMP_POOL);

    add_infinitesimal_inplace(H_Ht_H); /* adding 1e-6 to avoid division by 0 */
    mat_reciprocal_inplace(H_Ht_H);
    mat_elementwise_prod_inplace(W_H, H_Ht_H);
    mat_scalar_mult_inplace(W_H, beta);
    mat_add_scalar_inplace(W_H, 1 - beta);

    H_updated = mat_elementwise_prod(H, W_H, REGULAR_ALLOC);
    pool_free_all(TEMP_POOL); /* clean temporary memory */

    return H_updated;
}

int checkConvergence(MatrixPtr H, MatrixPtr H_updated, double epsilon){
    int isConverged;

    /* setting H = H_updated - H to check convergence */
    /* note -- H is being overridden (it is no longer needed) */
    mat_scalar_mult_inplace(H, -1);
    mat_add_inplace(H, H_updated);
    isConverged = mat_norm_sq(H) < epsilon; 

    return isConverged;
}

MatrixPtr getResultH(MatrixPtr H, MatrixPtr W){
    const double epsilon = 1e-4;
    const int maxIter = 300;
    int t;
    MatrixPtr H_updated;

    /* run maxiter times */
    for (t = 0; t < maxIter; t++){
        /* calculate new H according to the equation */
        H_updated = updateH(H, W);

        if (checkConvergence(H, H_updated, epsilon)){
            free_matrix(H);
            /* register H_updated to a given pool */
            pool_register_choice(MAIN_POOL, H_updated);
            return H_updated;
        }

        free_matrix(H);
        H = H_updated;
    }
    /* register H to a given pool */
    /* 
    this is required because H was not created in the pool 
    in order to free it each iteration */
    pool_register_choice(MAIN_POOL, H);
    return H;
}

MatrixPtr getSimilarityMatrix(MatrixPtr X){
    MatrixPtr A, diffVector;
    int i, j;
    double val;

    if (X==NULL) return NULL;
    
    A = create_matrix(X->m, X->m, MAIN_POOL);
    if (!A) return NULL;

    for (i = 0; i < X->m; i++)
    {
        for (j = 0; j < X->m; j++)
        {
            if (i == j){
                mat_set(A, i, j, 0); /* set 0 on the diagonal */
            }
            else{
                /* for entries that sits off the diagonal, compute row diff according to instructions for calculating A*/
                diffVector = get_row_diff(X, i, j, TEMP_POOL);
                if (!diffVector) return NULL;
                val = exp(-0.5 * mat_norm_sq(diffVector)); /* set value to be e^(0.5*||xi-xj||^2) */
                mat_set(A, i, j, val);
            }
        }
    }
    pool_free_all(TEMP_POOL); /* clean temporary memory */
    return A;
}

MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A)
{
    int i;
    double value;
    MatrixPtr D, sumVector;

    D = create_matrix(A->m, A->m, MAIN_POOL);
    if (!D) return NULL;

    /* sumVector is a col vector */
    /* sum of the columns */
    sumVector = sum_axis_0(A, TEMP_POOL);
    if (!sumVector) return NULL;

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
    /* taking inverse root on the diagonal to find D^-0.5 */
    diagonal_power_inplace(D, -0.5);
    /* performing the required matrix multiplication (D^-0.5)A(D^-0.5) */
    tmp = mat_dot_diagonal_left(D, A, TEMP_POOL);    
    W = mat_dot_diagonal_right(tmp, D, MAIN_POOL);
    pool_free_all(TEMP_POOL);
    return W;
}

/* Do not compile for python module */
#ifndef PYTHON_BUILD

int validity_check(int args_len, char *goal){
    if(args_len != 3) goto validity_error;
    if(strcmp("sym", goal) != 0) goto validity_error;
    if(strcmp("ddg", goal) != 0) goto validity_error;
    if(strcmp("norm", goal) != 0) goto validity_error;
    return 1;
    validity_error:
        return 0;
}

int main(int argc, char *argv[]){
    char *goal;
    FILE *file;
    MatrixPtr X, A, D, W;
    init_pools(); /* initializes memory pools */
    /* checks validity of input (using ? in case argc==1)*/
    if(validity_check(argc, goal = (argc > 1) ? argv[1] : " ") == 0 || !(file = fopen(argv[2], "r"))) goto validity_error;
    if(!(X = parse_matrix_from_stream(file))) goto free_and_exit; /* checks if X is null after parse_matrix_from_stream */

    if(!(A = getSimilarityMatrix(X))) goto free_and_exit; /* checks if A is null after getSimilarityMatrix */
    if (strcmp("sym", goal) == 0) print_matrix(A); /* print similarity matrix if goal="sym" */

    if(!(D = getDiagonalDegreeMatrix(A))) goto free_and_exit; /* checks if D is null after getDiagonalDegreeMatrix */
    if (strcmp("ddg", goal) == 0) print_matrix(D);/* print diagonal degree matrix if goal="ddg" */

    if(!(W = getNormalizedSimilarityMatrix(A, D))) goto free_and_exit; /* checks if W is null after getNormalizedSimilarityMatrix */
    if (strcmp("norm", goal) == 0) print_matrix(W); /* print normalized similarity matrix if goal="norm" */

    fclose(file);
    destroy_pools();
    return 0; /* Success */

    free_and_exit:
        fclose(file);
        validity_error: /* skips fclose file because file wasn't opened when the proggram failed*/
            ERROR_PRINT();
            destroy_pools();
            return ERROR_CODE;
}

#endif