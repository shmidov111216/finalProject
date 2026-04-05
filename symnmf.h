#ifndef SYMNMF_H
#define SYMNMF_H
#define INIT_CAP 32
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix_util.h"

MatrixPtr updateH(MatrixPtr H, MatrixPtr W);
MatrixPtr getResultH(MatrixPtr H, MatrixPtr W);
MatrixPtr getSimilarityMatrix(MatrixPtr X);
MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A);
MatrixPtr getNormalizedSimilarityMatrix(MatrixPtr A, MatrixPtr D);

#endif