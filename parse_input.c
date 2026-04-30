#include "parse_input.h"

int grow_line(char **line, size_t *cap, size_t len, FILE *fp){
    while (len == *cap - 1 && (*line)[len - 1] != '\n') /* line was cut, need more space */
    {
        char *tmp;

        *cap *= 2;                  /* double line buffer size */
        tmp = realloc(*line, *cap); /* resize line buffer */
        if (!tmp)
            return FAIL; /* allocation failed */

        *line = tmp; /* use new buffer */

        if (!fgets(*line + len, (int)(*cap - len), fp)) /* keep reading same line */
            break;

        len = strlen(*line); /* update current line length */
    }
    return SUCCESS;
}

int ensure_capacity(double **buffer, int *cap, int used){
    if (used >= *cap) /* numeric buffer is full */
    {
        double *tmp;

        *cap *= 2;                                       /* double numeric capacity */
        tmp = realloc(*buffer, (*cap) * sizeof(double)); /* resize numbers array */
        if (!tmp)
            return FAIL;

        *buffer = tmp; /* use resized array */
    }
    return SUCCESS; 
}

int parse_line(char *line, double **buffer, int *used, int *cap){
    char *token;
    int col_count = 0;
    token = strtok(line, ", \t\n"); /* first number in line */

    while (token)
    {
        if (!ensure_capacity(buffer, cap, *used))
            return -1; /* failed growing numeric array */

        (*buffer)[(*used)++] = atof(token); /* save parsed number */
        col_count++;                        /* count values in this row */
        token = strtok(NULL, ", \t\n");     /* next number */
    }
    return col_count; /* how many columns this row had */
}

MatrixPtr parse_matrix_from_stream(FILE *fp){
    int rows = 0, cols = -1, total_used = 0, col_count, total_cap = INIT_CAP;
    size_t line_cap = 1024, len;
    char *line = malloc(line_cap); /* buffer for reading text rows */
    MatrixPtr X;
    double *buffer = malloc(INIT_CAP * sizeof(double)); /* stores all matrix values */

    if (!line || !buffer) goto memory_error; /* initial allocation failed */

    while (fgets(line, (int)line_cap, fp)) /* read one row at a time */
    {
        len = strlen(line);

        if (!grow_line(&line, &line_cap, len, fp)) goto memory_error; /* failed resizing line buffer */

        col_count = parse_line(line, &buffer, &total_used, &total_cap); /* parse current row */
        if (col_count < 0) goto memory_error; /* failed parsing numeric values */
        if (cols == -1) cols = col_count; /* first row sets column count */
        rows++; /* valid row counted */
    }
    free(line); /* text buffer no longer needed */

    X = create_matrix(rows, cols, MAIN_POOL); /* create final matrix */
    if (!X) goto memory_error;

    X->data = buffer; /* attach parsed values */
    return X;

    memory_error:
    free(line); /* free if allocated */
    free(buffer);
    return NULL; /* signal failure */
}