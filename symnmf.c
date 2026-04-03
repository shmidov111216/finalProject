#include "symnmf.h"

/* -------- Core Algorithm ---------- */

MatrixPtr updateH(MatrixPtr H, MatrixPtr W)
{
    double beta = 0.5;
    MatrixPtr H_updated, W_H, Ht, H_Ht, H_Ht_H;

    W_H = mat_dot(W, H);
    Ht = mat_transpose(H);
    H_Ht = mat_dot(H, Ht);
    H_Ht_H = mat_dot(H_Ht, H);
    // fill H_Ht_H zeroes

    replace_zeroes(H_Ht_H);
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

MatrixPtr getSimilarityMatrix(MatrixPtr X)
{
    MatrixPtr A = create_matrix(X->m, X->m);

    if (!A)
    {
        return NULL;
    }

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
                diffVector = get_row_diff(X, i, j);
                val = exp(-0.5 * mat_norm_sq(diffVector));
                mat_set(A, i, j, val);
                free_matrix(diffVector);
            }
        }
    }
    return A;
}

MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A)
{
    int i;
    MatrixPtr D = create_matrix(A->m, A->m);
    double value;

    if (!D)
    {
        return NULL;
    }

    // sumVector is a col vector
    MatrixPtr sumVector = sum_axis_0(A);

    if (!sumVector)
    {
        return NULL;
    }

    for (i = 0; i < D->m; i++)
    {
        value = mat_get(sumVector, i, 0);
        mat_set(D, i, i, value);
    }
    free_matrix(sumVector);
    return D;
}

MatrixPtr getNormalizedSimilarityMatrix(MatrixPtr A, MatrixPtr D)
{
    MatrixPtr W, tmp;
    // create D^-1/2
    //printf("\nentering getNormalizedSimilarityMatrix\n");
    
    diagonal_power_inplace(D, -0.5);
    //printf("\ndiagonal_power_inplace\n");
    
    tmp = mat_dot_diagonal_left(D, A);
    //printf("\nmat_dot_diagonal_left(D, A)\n");
    
    W = mat_dot_diagonal_right(tmp, D);
    //printf("\nmat_dot(tmp, D)\n");
    free_matrix(tmp);
    // check if W is null
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

    MatrixPtr A = create_matrix(rows, cols);
    CHECK_MATRIX_ALLOC(A);

    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            A->data[i][j] = buffer[bufferCnt++];
        }
    }
    free(buffer);

    return A;
}

int main(int argc, char *argv[]){
    char *file_name = argv[2];
    char *goal = argv[1];
    //printf("opening file\n");
    
    FILE *file = fopen(file_name, "r");
    
    //printf("before create matrix\n");
    MatrixPtr X = parse_matrix_from_stream(file);
    
    //printf("created matrix success\n");
    MatrixPtr A, D, W;

    A = getSimilarityMatrix(X);
    
    if (strcmp("sym", goal) == 0)
        ;
    // print_matrix(A);
    D = getDiagonalDegreeMatrix(A);
    /*
    if (strcmp("ddg", goal) == 0)
        print_matrix(D);
    */

    W = getNormalizedSimilarityMatrix(A, D);
    W = W;
    /*
    if (strcmp("norm", goal) == 0)
        print_matrix(W);
    */

    //printf("\nSuccess\n");
}