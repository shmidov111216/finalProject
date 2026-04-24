#ifndef SYMNMF_H
#define SYMNMF_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix_util.h"
#include "parse_input.h"

MatrixPtr updateH(MatrixPtr H, MatrixPtr W);
MatrixPtr getResultH(MatrixPtr H, MatrixPtr W);
MatrixPtr getSimilarityMatrix(MatrixPtr X);
MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A);
MatrixPtr getNormalizedSimilarityMatrix(MatrixPtr A, MatrixPtr D);

#endif