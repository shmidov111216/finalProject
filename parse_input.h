#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_util.h"
#define INIT_CAP 16

int grow_line(char **line, size_t *cap, size_t len, FILE *fp);
int ensure_capacity(double **buffer, int *cap, int used);
int parse_line(char *line, double **buffer, int *used, int *cap);
MatrixPtr parse_matrix_from_stream(FILE *fp);
