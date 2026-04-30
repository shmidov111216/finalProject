#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_util.h"
#define INIT_CAP 16

/*
Ensures the line buffer is large enough to hold a full line from the stream.
If the buffer is too small, its capacity is repeatedly doubled until it fits.

Input:
  line - dynamic character buffer
  cap  - current buffer capacity (in bytes, including '\0')
  len  - current length of the string in the buffer
  fp   - input stream to read more data from

Output:
  Returns 1 on success, 0 on allocation failure.
 */
int grow_line(char **line, size_t *cap, size_t len, FILE *fp);

/*
  Ensures the buffer has enough space for one more double.
  If the buffer is full, its capacity is doubled using realloc.
  Input:
    buffer - pointer to a dynamic array of doubles
    cap    - current capacity of the buffer
    used   - number of elements currently stored

  Output: Returns 1 on success, 0 if reallocation fails.
 */
int ensure_capacity(double **buffer, int *cap, int used);

/*
Parses a single line into doubles separated by commas or whitespace.
Parsed values are appended to the buffer.

Input:
  line    - input line (will be modified)
  buffer  - array to store parsed values
  used    - number of values already stored
  cap     - buffer capacity

Output:
  Returns the number of values parsed,
  or -1 on allocation failure.
*/
int parse_line(char *line, double **buffer, int *used, int *cap);

/*
Reads numeric data from a stream and builds a matrix. Each line becomes a row in the output matrix.
Input:
  fp - input stream (must have the same number of floats in each line)
Output:
  Returns the parsed matrix,
  or NULL on error.
*/
MatrixPtr parse_matrix_from_stream(FILE *fp);
