#include "symnmf.h"

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
    // fill H_Ht_H zeroes

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
    MatrixPtr A = create_matrix(X->m, X->m, MAIN_POOL);

    CHECK_MATRIX_ALLOC(A);

    MatrixPtr diffVector;
    int i, j;
    double val;

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
    MatrixPtr D = create_matrix(A->m, A->m, MAIN_POOL);
    //D = NULL;
    double value;

    CHECK_MATRIX_ALLOC(D);

    // sumVector is a col vector
    MatrixPtr sumVector = sum_axis_0(A, TEMP_POOL);
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
    //printf("\ndiagonal_power_inplace\n");
    //printf("\neeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n");
    
    tmp = mat_dot_diagonal_left(D, A, TEMP_POOL);
    //printf("\nmat_dot_diagonal_left(D, A)\n");
    
    W = mat_dot_diagonal_right(tmp, D, MAIN_POOL);
    //printf("\nmat_dot(tmp, D)\n");

    pool_free_all(TEMP_POOL);
    return W;
}

MatrixPtr parse_matrix_from_stream(FILE *fp)
{
    char *line = NULL;
    size_t len = 0;

    int rows_cap = INIT_CAP;
    int cols = -1;
    int rows = 0;
    int bufferCnt = 0;

    double *buffer = malloc(rows_cap * INIT_CAP * sizeof(double));
    CHECK_MATRIX_ALLOC(buffer);

    int total_cap = rows_cap * INIT_CAP;
    int total_used = 0;

    while (getline(&line, &len, fp) != -1)
    {
        int col_count = 0;

        char *token = strtok(line, ", \t\n");
        while (token)
        {
            if (total_used >= total_cap)
            {
                total_cap *= 2;
                buffer = realloc(buffer, total_cap * sizeof(double));
                CHECK_MATRIX_ALLOC(buffer);
            }

            buffer[total_used++] = atof(token);
            col_count++;

            token = strtok(NULL, ", \t\n");
        }

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

    free(line);

    MatrixPtr X = create_matrix(rows, cols, MAIN_POOL);
    CHECK_MATRIX_ALLOC(X);

    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            MAT(X, i, j) = buffer[bufferCnt++];
        }
    }
    free(buffer);

    return X;
}

int main(int argc, char *argv[]){
    char *file_name = argv[2];
    char *goal = argv[1];
    //goal = goal;
    FILE *file = fopen(file_name, "r");
    init_pools();

    printf("Hello not ok ");

    MatrixPtr X = parse_matrix_from_stream(file);
    CHECK_FREE_AND_EXIT(X);

    MatrixPtr A, D, W;

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
    
}