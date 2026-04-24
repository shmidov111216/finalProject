#include "parse_input.h"

int grow_line(char **line, size_t *cap, size_t len, FILE *fp)
{
    while (len == *cap - 1 && (*line)[len - 1] != '\n')
    {
        char *tmp;

        *cap *= 2;
        tmp = realloc(*line, *cap);
        if (!tmp)
            return 0;

        *line = tmp;

        if (!fgets(*line + len, (int)(*cap - len), fp))
            break;

        len = strlen(*line);
    }
    return 1;
}

int ensure_capacity(double **buffer, int *cap, int used)
{
    if (used >= *cap)
    {
        double *tmp;

        *cap *= 2;
        tmp = realloc(*buffer, (*cap) * sizeof(double));
        if (!tmp)
            return 0;

        *buffer = tmp;
    }
    return 1;
}

int parse_line(char *line, double **buffer, int *used, int *cap)
{
    char *token;
    int col_count = 0;
    token = strtok(line, ", \t\n");

    while (token)
    {
        if (!ensure_capacity(buffer, cap, *used))
            return -1;

        (*buffer)[(*used)++] = atof(token);
        col_count++;
        token = strtok(NULL, ", \t\n");
    }
    return col_count;
}

MatrixPtr parse_matrix_from_stream(FILE *fp){
    size_t line_cap = 1024;
    char *line = malloc(line_cap);
    MatrixPtr X;

    double *buffer = malloc(INIT_CAP * sizeof(double));

    int rows = 0, cols = -1;
    int total_used = 0, total_cap = INIT_CAP;

    if (!line || !buffer) goto memory_error;

    while (fgets(line, (int)line_cap, fp))
    {
        size_t len = strlen(line);
        int col_count;

        if (!grow_line(&line, &line_cap, len, fp))
            goto memory_error;

        col_count = parse_line(line, &buffer, &total_used, &total_cap);
        if (col_count < 0) goto memory_error;

        if (col_count == 0) continue;

        if (cols == -1) cols = col_count;
        else if (cols != col_count) goto memory_error;

        rows++;
    }
    free(line);
    
    X = create_matrix(rows, cols, MAIN_POOL);
    if (!X) goto memory_error;
    X->data = buffer; 
    return X;

    memory_error:
    free(line);
    free(buffer);
    return NULL;
}